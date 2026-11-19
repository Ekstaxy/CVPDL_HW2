"""Normalization-only TTA: BN-Adapt (test-batch statistics) and Robust BN (EMA of statistics, from RoTTA)."""
from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import force_bn_batch_stats, replace_bn


class BNAdapt:
    """BN-Adapt / "norm_test" (Schneider et al. 2020; Nado et al. 2020).

    Every BN layer normalizes with the statistics of the current test batch; nothing is learned.
    """

    def __init__(self, model):
        self.model = model.eval()
        self.model.requires_grad_(False)
        force_bn_batch_stats(self.model)
        self.last_logits = None

    @torch.no_grad()
    def __call__(self, x):
        out = self.model(x)
        self.last_logits = out
        return out.argmax(1)


class RobustBN2d(nn.Module):
    """Robust BN from RoTTA (Yuan et al., CVPR 2023). Port of `RobustBN2d` in the official code.

    Keeps global statistics initialized from the source model. In train mode, they are updated with the
    current batch by an EMA with momentum alpha and the *updated* statistics are used to normalize.
    In eval mode, the stored statistics are used as they are.
    """

    def __init__(self, bn: nn.BatchNorm2d, momentum: float):
        super().__init__()
        self.num_features, self.momentum, self.eps = bn.num_features, momentum, bn.eps
        self.register_buffer("source_mean", deepcopy(bn.running_mean))
        self.register_buffer("source_var", deepcopy(bn.running_var))
        self.weight = deepcopy(bn.weight)
        self.bias = deepcopy(bn.bias)

        # used by AdaptiveRobustBN
        self.probe = False      # probe mode: measure discrepancy, normalize with stored stats, no update
        self.last_kl = None     # symmetric KL(batch stats || stored stats), mean over channels
        self.adaptive_gain = None  # per-layer mode: momentum = clip(gain * own KL, lo, hi)
        self.alpha_range = None
        self.fused = False      # normalize + affine as one per-channel x * scale + shift (same math, faster)
        self.fast_eval = False  # eval mode: normalize with F.batch_norm (stored statistics)

    def discrepancy(self, b_mean, b_var):
        ev, em, eps = self.source_var, self.source_mean, self.eps
        kl = 0.5 * ((b_var + eps) / (ev + eps) + (ev + eps) / (b_var + eps) - 2
                    + (b_mean - em) ** 2 * (1 / (ev + eps) + 1 / (b_var + eps)))
        return kl.mean()

    def forward(self, x):
        if self.probe:
            with torch.no_grad():
                b_var, b_mean = torch.var_mean(x, dim=[0, 2, 3], unbiased=False)
                self.last_kl = self.discrepancy(b_mean, b_var)
            mean, var = self.source_mean.view(1, -1, 1, 1), self.source_var.view(1, -1, 1, 1)
        elif self.training:
            b_var, b_mean = torch.var_mean(x, dim=[0, 2, 3], unbiased=False)
            if self.adaptive_gain is not None:
                with torch.no_grad():
                    self.last_kl = self.discrepancy(b_mean, b_var)
                lo, hi = self.alpha_range
                self.momentum = float((self.adaptive_gain * self.last_kl).clamp(lo, hi))
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean, self.source_var = mean.detach().clone(), var.detach().clone()
            mean, var = mean.view(1, -1, 1, 1), var.view(1, -1, 1, 1)
        else:
            if self.fast_eval:  # one native op instead of ~8 small ones (same math up to rounding, ~20% faster run)
                return F.batch_norm(x, self.source_mean, self.source_var, self.weight, self.bias, False, 0.0, self.eps)
            mean, var = self.source_mean.view(1, -1, 1, 1), self.source_var.view(1, -1, 1, 1)
        if self.fused:
            scale = self.weight / torch.sqrt(var.flatten() + self.eps)
            shift = self.bias - mean.flatten() * scale
            return torch.addcmul(shift.to(x.dtype).view(1, -1, 1, 1), x, scale.to(x.dtype).view(1, -1, 1, 1))
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class RobustBN:
    """Robust BN alone (no gradient updates), alpha = 0.05 as in RoTTA.

    Each test batch first updates the EMA statistics and is then normalized with them.
    """

    ALPHA = 0.05

    def __init__(self, model):
        model.requires_grad_(False)
        self.model = replace_bn(model, lambda bn: RobustBN2d(bn, self.ALPHA)).cuda().train()
        self.last_logits = None

    @torch.no_grad()
    def __call__(self, x):
        out = self.model(x)
        self.last_logits = out
        return out.argmax(1)


class AdaptiveRobustBN:
    """Robust BN whose EMA momentum grows with the detected distribution shift.

    shift signal s_t = mean symmetric KL between the current batch statistics and the stored EMA statistics,
    measured (before any update) on the first N_DETECT BN layers (stem + layer1). Shallow layers react to the
    corruption (domain) but hardly to the class composition of the batch, so label shift does not trigger it.

        alpha_t = clip(GAIN * s_t, ALPHA_MIN, ALPHA_MAX)

    Large shift -> large alpha (stats jump towards the new domain); as the EMA catches up, s_t shrinks and
    alpha decays back to ALPHA_MIN on its own. DETECT = "per_layer" instead lets every layer use its own KL.
    The decision only uses the batch being predicted and the past (allowed by the HW2 rules).
    """

    ALPHA_MIN = 0.05
    ALPHA_MAX = 0.5
    GAIN = 1.0
    DETECT = "shallow"   # "shallow" (global signal from stem + layer1) | "per_layer"
    N_DETECT = 11        # bn1 + the 10 BN layers of layer1

    def __init__(self, model, alpha_min=None, alpha_max=None, gain=None, detect=None):
        self.alpha_min = self.ALPHA_MIN if alpha_min is None else alpha_min
        self.alpha_max = self.ALPHA_MAX if alpha_max is None else alpha_max
        self.gain = self.GAIN if gain is None else gain
        self.detect = self.DETECT if detect is None else detect
        model.requires_grad_(False)
        self.model = replace_bn(model, lambda bn: RobustBN2d(bn, self.alpha_min)).cuda().train()
        self.layers = [m for m in self.model.modules() if isinstance(m, RobustBN2d)]
        if self.detect == "per_layer":
            for l in self.layers:
                l.adaptive_gain, l.alpha_range = self.gain, (self.alpha_min, self.alpha_max)
        self.alpha_hist, self.signal_hist = [], []
        self.last_logits = None

    def _shallow_signal(self, x):
        m = self.model
        for l in self.layers:
            l.probe = True
        m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(x)))))
        for l in self.layers:
            l.probe = False
        return torch.stack([l.last_kl for l in self.layers[: self.N_DETECT]]).mean()

    @torch.no_grad()
    def __call__(self, x):
        if self.detect == "shallow":
            s = self._shallow_signal(x)
            alpha = float((self.gain * s).clamp(self.alpha_min, self.alpha_max))
            for l in self.layers:
                l.momentum = alpha
            self.signal_hist.append(float(s)); self.alpha_hist.append(alpha)
        out = self.model(x)
        if self.detect == "per_layer":
            self.alpha_hist.append(float(np.mean([l.momentum for l in self.layers])))
        self.last_logits = out
        return out.argmax(1)
