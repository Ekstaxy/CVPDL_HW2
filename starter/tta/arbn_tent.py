"""Adaptive Robust BN + TENT: entropy minimization on the BN affine parameters (gamma, beta) on top of arbn.

Per batch:
  1. probe stem + layer1 -> shift signal s_t -> momentum alpha_t = clip(GAIN * s_t, ALPHA_MIN, ALPHA_MAX)
  2. forward (train mode): EMA statistics updated with alpha_t, logits computed with the updated statistics
  3. loss = mean entropy of the (optionally filtered) predictions; one optimizer step on gamma / beta
  4. prediction = logits of step 2 (as TENT), or of a second forward after the update (PREDICT_AFTER)

Knobs (class constants, or constructor kwargs):
  LR, OPTIM          learning rate / "sgd" (momentum 0.9, nesterov) or "adam"
  E0                 entropy filter: only samples with entropy < E0 * ln(K) contribute (None = no filter)
  SRC_MOMENTUM       after each step pull gamma/beta back to the source values: p = m * p + (1 - m) * p_src
                     (ROID-style weight ensembling; None = off)
  UPDATE_LAYERS      "all" | "deep" (layer3 + layer4 + ...) | "shallow" (stem + layer1 + layer2)
  PREDICT_AFTER      predict with the model after the update (extra forward, uses the stored statistics)
  LAME_K             output-only LAME (Boudiaf et al., CVPR 2022): Laplacian smoothing of the predictions over a
                     kNN graph of the batch features, exploits that a non-i.i.d. batch holds few classes
                     (None = off). Does not affect the adaptation itself.
  T3A_K              output-only T3A class prototypes (support set of FILTER_K per class), applied before LAME
                     (None = off).
"""
import math

import torch
import torch.nn.functional as F

from .common import entropy
from .norm import AdaptiveRobustBN, RobustBN2d
from .t3a import T3A


@torch.no_grad()
def lame(logits, feats, k=5, iters=10, power=False, time_w=0.0, time_tau=4):
    """LAME with a symmetric kNN affinity on L2-normalized features (port of the reference implementation).

    power = True   square root of the (non-negative, post-ReLU) features before the cosine kNN search: damps the few
                   dominant channels so that neighbours are chosen on more of the feature vector.
    time_w > 0     temporal affinity added to the kNN graph: time_w for every pair of images at most time_tau
                   positions apart in the batch. The stream is non-i.i.d. (consecutive images come from a few classes),
                   so stream neighbours are likely to share the label. Uses only the order inside the current batch.
    """
    f = feats.float()
    if power:
        f = f.clamp_min(0).sqrt()
    f = F.normalize(f, dim=1)
    sim = f @ f.T
    knn = sim.topk(k + 1, dim=1).indices[:, 1:]
    W = torch.zeros_like(sim).scatter_(1, knn, 1.0)
    W = (W + W.T) / 2
    if time_w > 0:
        i = torch.arange(len(f), device=f.device)
        d = (i[:, None] - i[None, :]).abs()
        W = W + time_w * ((d <= time_tau) & (d > 0)).float()
    log_p = logits.float().log_softmax(1)
    Z = log_p.softmax(1)
    for _ in range(iters):
        Z = (log_p + W @ Z).softmax(1)
    return Z


class AdaptiveRBNTent(AdaptiveRobustBN):
    LR = 0.1
    OPTIM = "sgd"
    E0 = None
    SRC_MOMENTUM = None
    UPDATE_LAYERS = "all"
    PREDICT_AFTER = False
    LAME_K = 5
    T3A_K = None
    LAME_POW = False     # LAME graph: square-root feature transform before the kNN search
    LAME_TIME = 0.0      # LAME graph: weight of the temporal affinity (0 = off)
    LAME_TAU = 4         # LAME graph: temporal affinity links images at most this many stream positions apart
    NUM_CLASSES = 20

    def __init__(self, model, lr=None, optim=None, e0="default", src_momentum="default", update_layers=None,
                 predict_after=None, lame_k="default", t3a_k="default", lame_pow=None, lame_time=None, lame_tau=None,
                 **arbn_kwargs):
        super().__init__(model, **arbn_kwargs)
        self.lame_kw = dict(power=self.LAME_POW if lame_pow is None else lame_pow,
                            time_w=self.LAME_TIME if lame_time is None else lame_time,
                            time_tau=self.LAME_TAU if lame_tau is None else lame_tau)
        self.lr = self.LR if lr is None else lr
        self.optim = self.OPTIM if optim is None else optim
        self.e0 = self.E0 if e0 == "default" else e0
        self.src_momentum = self.SRC_MOMENTUM if src_momentum == "default" else src_momentum
        self.update_layers = self.UPDATE_LAYERS if update_layers is None else update_layers
        self.predict_after = self.PREDICT_AFTER if predict_after is None else predict_after
        self.lame_k = self.LAME_K if lame_k == "default" else lame_k
        self.t3a_k = self.T3A_K if t3a_k == "default" else t3a_k
        self.t3a = T3A(self.model.fc, self.t3a_k, self.NUM_CLASSES) if self.t3a_k is not None else None
        self._feats = None
        self.model.avgpool.register_forward_hook(lambda m, i, o: setattr(self, "_feats", o.flatten(1)))

        shallow_prefixes = ("bn1", "layer1.", "layer2.")
        params = []
        for name, m in self.model.named_modules():
            if not isinstance(m, RobustBN2d):
                continue
            is_shallow = name.startswith(shallow_prefixes)
            use = self.update_layers == "all" or (self.update_layers == "shallow") == is_shallow
            m.weight.requires_grad_(use)
            m.bias.requires_grad_(use)
            if use:
                params += [m.weight, m.bias]
        self.params = params
        self.src_params = [p.detach().clone() for p in params]
        if self.optim == "adam":
            self.optimizer = torch.optim.Adam(params, lr=self.lr, betas=(0.9, 0.999))
        else:
            self.optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9, nesterov=True)
        self.skipped = 0

    def __call__(self, x):
        with torch.no_grad():
            s = self._shallow_signal(x)
            alpha = float((self.gain * s).clamp(self.alpha_min, self.alpha_max))
            for l in self.layers:
                l.momentum = alpha
            self.signal_hist.append(float(s)); self.alpha_hist.append(alpha)

        self.model.train()
        with torch.enable_grad():
            out = self.model(x)
            ent = entropy(out)
            if self.e0 is not None:
                mask = ent < self.e0 * math.log(self.NUM_CLASSES)
                loss = ent[mask].mean() if mask.any() else None
            else:
                loss = ent.mean()
            if loss is not None:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                self.skipped += 1

        if self.src_momentum is not None:
            with torch.no_grad():
                for p, p0 in zip(self.params, self.src_params):
                    p.mul_(self.src_momentum).add_(p0, alpha=1 - self.src_momentum)

        if self.predict_after:
            with torch.no_grad():
                self.model.eval()  # stored (already updated) statistics, no second statistics update
                out = self.model(x)
                self.model.train()
        out = out.detach()
        self.last_logits = out
        return self.finalize(out, self._feats.detach())

    def finalize(self, logits, feats):
        """Output-only stages: T3A prototypes (optional), then LAME (optional) -> labels."""
        if self.t3a is not None:
            logits = self.t3a(feats, logits)
        if self.lame_k is not None:
            return lame(logits, feats, self.lame_k, **self.lame_kw).argmax(1)
        return logits.argmax(1)


class AdaptiveRBNLame(AdaptiveRobustBN):
    """Adaptive Robust BN + output-only LAME (no gradient updates). Same arbn settings, LAME k = 5."""

    LAME_K = 5

    def __init__(self, model, lame_k="default", **arbn_kwargs):
        super().__init__(model, **arbn_kwargs)
        self.lame_k = self.LAME_K if lame_k == "default" else lame_k
        self._feats = None
        self.model.avgpool.register_forward_hook(lambda m, i, o: setattr(self, "_feats", o.flatten(1)))

    @torch.no_grad()
    def __call__(self, x):
        super().__call__(x)  # sets self.last_logits (arbn output) and self._feats
        if self.lame_k is not None:
            return lame(self.last_logits, self._feats, self.lame_k).argmax(1)
        return self.last_logits.argmax(1)
