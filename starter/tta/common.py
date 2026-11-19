"""Shared helpers for the TTA methods: (de)normalization, augmentations, losses, EMA, model surgery.

The harness feeds *normalized* tensors, while all reference augmentation pipelines (CoTTA / RoTTA / RMT /
ROID / TRIBE) operate on images in [0, 1]. `NormalizedAug` bridges the two.
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
IMG_SIZE = 160


# ----------------------------------------------------------------------------------------------------------
# augmentations (port of CoTTA's `get_tta_transforms`, used by CoTTA / RoTTA / RMT / ROID / TRIBE)
# ----------------------------------------------------------------------------------------------------------
class Clip(nn.Module):
    def __init__(self, lo=0.0, hi=1.0):
        super().__init__()
        self.lo, self.hi = lo, hi

    def forward(self, x):
        return x.clamp(self.lo, self.hi)


class GaussianNoise(nn.Module):
    def __init__(self, std=0.005):
        super().__init__()
        self.std = std

    def forward(self, x):
        return x + torch.randn_like(x) * self.std


class ColorJitterPro(T.ColorJitter):
    """ColorJitter + gamma, applied in a random order (same parameters for the whole batch, as in the original)."""

    def __init__(self, brightness, contrast, saturation, hue, gamma):
        super().__init__(brightness, contrast, saturation, hue)
        self.gamma = self._check_input(gamma, "gamma")

    def forward(self, img):
        for fn_id in torch.randperm(5):
            if fn_id == 0 and self.brightness is not None:
                img = TF.adjust_brightness(img, torch.empty(1).uniform_(*self.brightness).item())
            if fn_id == 1 and self.contrast is not None:
                img = TF.adjust_contrast(img, torch.empty(1).uniform_(*self.contrast).item())
            if fn_id == 2 and self.saturation is not None:
                img = TF.adjust_saturation(img, torch.empty(1).uniform_(*self.saturation).item())
            if fn_id == 3 and self.hue is not None:
                img = TF.adjust_hue(img, torch.empty(1).uniform_(*self.hue).item())
            if fn_id == 4 and self.gamma is not None:
                img = TF.adjust_gamma(img.clamp(1e-8, 1.0), torch.empty(1).uniform_(*self.gamma).item())
        return img


def get_tta_transforms(img_size=IMG_SIZE, gaussian_std=0.005, soft=False, padding_mode="edge", cotta_augs=True,
                       fill=0):
    """Port of `augmentations/transforms_cotta.get_tta_transforms` (CoTTA / RMT / ROID / RoTTA / TRIBE)."""
    tf = [
        Clip(0.0, 1.0),
        ColorJitterPro(brightness=[0.8, 1.2] if soft else [0.6, 1.4],
                       contrast=[0.85, 1.15] if soft else [0.7, 1.3],
                       saturation=[0.75, 1.25] if soft else [0.5, 1.5],
                       hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
                       gamma=[0.85, 1.15] if soft else [0.7, 1.3]),
        T.Pad(padding=img_size // 2, padding_mode=padding_mode),
        T.RandomAffine(degrees=[-8, 8] if soft else [-15, 15], translate=(1 / 16, 1 / 16),
                       scale=(0.95, 1.05) if soft else (0.9, 1.1), shear=None,
                       interpolation=T.InterpolationMode.BILINEAR, fill=fill),
    ]
    if cotta_augs:
        tf += [T.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
               T.CenterCrop(size=img_size), T.RandomHorizontalFlip(p=0.5), GaussianNoise(gaussian_std), Clip(0.0, 1.0)]
    else:
        tf += [T.CenterCrop(size=img_size), T.RandomHorizontalFlip(p=0.5), Clip(0.0, 1.0)]
    return T.Compose(tf)


class NormalizedAug:
    """Apply a [0,1]-space augmentation to a normalized batch: denorm -> aug -> renorm."""

    def __init__(self, transform):
        self.t = transform

    def __call__(self, x):
        mean, std = MEAN.to(x.device), STD.to(x.device)
        return (self.t(x * std + mean) - mean) / std


# ----------------------------------------------------------------------------------------------------------
# losses
# ----------------------------------------------------------------------------------------------------------
def entropy(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def softmax_entropy(x, x_ema):
    """Cross-entropy of student x w.r.t. teacher soft labels x_ema."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


def symmetric_cross_entropy(x, x_ema, alpha=0.5):
    return -(1 - alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - alpha * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)


def soft_likelihood_ratio(logits, clip=0.99, eps=1e-5):
    p = logits.softmax(1).clamp(0.0, clip)
    return -(p * torch.log(p / (1.0 - p) + eps)).sum(1)


# ----------------------------------------------------------------------------------------------------------
# model helpers
# ----------------------------------------------------------------------------------------------------------
def build_ema(model):
    ema = deepcopy(model)
    for p in ema.parameters():
        p.detach_()
    return ema


@torch.no_grad()
def ema_update(target, source, momentum, update_all=True):
    """target = momentum * target + (1 - momentum) * source (parameters only)."""
    for pt, ps in zip(target.parameters(), source.parameters()):
        if update_all or pt.requires_grad:
            pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)


def force_bn_batch_stats(model):
    """Make every BatchNorm2d normalize with the current test batch statistics (in train and eval mode)."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None


def bn_affine_params(model):
    params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for n, p in m.named_parameters(recurse=False):
                if n in ("weight", "bias") and p.requires_grad:
                    params.append(p)
    return params


def get_submodule(model, name):
    for part in name.split("."):
        model = getattr(model, part)
    return model


def set_submodule(model, name, value):
    *parents, last = name.split(".")
    for part in parents:
        model = getattr(model, part)
    setattr(model, last, value)


def replace_bn(model, factory):
    """Replace every BatchNorm2d in `model` by factory(bn_layer)."""
    names = [n for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]
    for n in names:
        set_submodule(model, n, factory(get_submodule(model, n)))
    return model


def sgd(params, lr):
    """SGD as configured in the TENT / CoTTA / RMT / ROID ImageNet-C configs (momentum 0.9, nesterov)."""
    return torch.optim.SGD(params, lr=lr, momentum=0.9, dampening=0.0, weight_decay=0.0, nesterov=True)


def adam(params, lr):
    return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
