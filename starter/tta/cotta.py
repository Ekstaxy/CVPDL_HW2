"""CoTTA: Continual Test-Time Domain Adaptation (Wang et al., CVPR 2022).

ImageNet-C config (test-time-adaptation benchmark repo): SGD lr 0.01 (momentum 0.9, nesterov), teacher EMA
0.999, stochastic restore probability 0.001, augmentation threshold AP 0.1, 32 augmentations,
symmetric ("imagenet") cross-entropy. All parameters are trained; BN uses test-batch statistics.
Prediction = teacher output (augmentation-averaged if the source model is unconfident).
"""
import torch

from .common import NormalizedAug, build_ema, ema_update, force_bn_batch_stats, get_tta_transforms, sgd, \
    symmetric_cross_entropy


class CoTTA:
    LR = 0.01
    MT = 0.999
    RST = 0.001
    AP = 0.1
    N_AUG = 32

    def __init__(self, model):
        self.model = model.eval()
        force_bn_batch_stats(self.model)
        self.model.requires_grad_(True)
        self.optimizer = sgd(self.model.parameters(), self.LR)
        self.model_ema = build_ema(self.model)
        self.model_anchor = build_ema(self.model)
        self.source_params = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        self.transform = NormalizedAug(get_tta_transforms())
        self.n_aug_batches = 0
        self.last_logits = None

    @torch.enable_grad()
    def __call__(self, x):
        out = self.model(x)
        with torch.no_grad():
            anchor_prob = self.model_anchor(x).softmax(1).max(1)[0]
            if anchor_prob.mean(0) < self.AP:
                self.n_aug_batches += 1
                out_ema = torch.stack([self.model_ema(self.transform(x)) for _ in range(self.N_AUG)]).mean(0)
            else:
                out_ema = self.model_ema(x)
        loss = symmetric_cross_entropy(out, out_ema).mean(0)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        ema_update(self.model_ema, self.model, self.MT)
        # stochastic restore of weights / biases towards the source model
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n.endswith("weight") or n.endswith("bias"):
                    mask = (torch.rand(p.shape, device=p.device) < self.RST).float()
                    p.data = self.source_params[n] * mask + p.data * (1.0 - mask)
        self.last_logits = out_ema
        return out_ema.argmax(1)
