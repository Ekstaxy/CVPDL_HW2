"""ROID: Universal Test-time Adaptation through Weight Ensembling, Diversity Weighting, and Prior Correction
(Marsden, Döbler, Yang, WACV 2024). Port of the official implementation.

ImageNet-C config: SGD lr 2.5e-4 (momentum 0.9, nesterov), BN affine parameters only, BN uses test-batch
statistics, weighting + prior correction + consistency all on, momentum_src 0.99, momentum_probs 0.9,
temperature 1/3. Loss = weighted soft-likelihood-ratio + weighted symmetric-CE consistency to an augmented view.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import NormalizedAug, bn_affine_params, build_ema, ema_update, entropy, force_bn_batch_stats, \
    get_tta_transforms, sgd, soft_likelihood_ratio, symmetric_cross_entropy


class ROID:
    LR = 2.5e-4
    MOMENTUM_SRC = 0.99
    MOMENTUM_PROBS = 0.9
    TEMPERATURE = 1 / 3
    BATCH_SIZE = 64
    NUM_CLASSES = 20

    def __init__(self, model):
        self.model = model.eval()
        self.model.requires_grad_(False)
        force_bn_batch_stats(self.model)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
        self.optimizer = sgd(bn_affine_params(self.model), self.LR)
        self.src_model = build_ema(self.model)
        self.class_probs_ema = torch.full((self.NUM_CLASSES,), 1 / self.NUM_CLASSES, device="cuda")
        self.transform = NormalizedAug(get_tta_transforms(padding_mode="reflect", cotta_augs=False))
        self.last_logits = None

    @torch.enable_grad()
    def __call__(self, x):
        out = self.model(x)
        with torch.no_grad():
            p = out.softmax(1)
            w_div = 1 - F.cosine_similarity(self.class_probs_ema.unsqueeze(0), p, dim=1)
            w_div = (w_div - w_div.min()) / (w_div.max() - w_div.min())
            mask = w_div < w_div.mean()
            w_cert = -entropy(out)
            w_cert = (w_cert - w_cert.min()) / (w_cert.max() - w_cert.min())
            weights = torch.exp(w_div * w_cert / self.TEMPERATURE)
            weights[mask] = 0.0
            self.class_probs_ema = self.MOMENTUM_PROBS * self.class_probs_ema + (1 - self.MOMENTUM_PROBS) * p.mean(0)
        loss = (soft_likelihood_ratio(out) * weights)[~mask].sum() / self.BATCH_SIZE
        out_aug = self.model(self.transform(x[~mask]))
        loss = loss + (symmetric_cross_entropy(out_aug, out[~mask]) * weights[~mask]).sum() / self.BATCH_SIZE
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # weight ensembling with the frozen source model
        ema_update(self.model, self.src_model, self.MOMENTUM_SRC, update_all=False)
        with torch.no_grad():
            # prior correction (applied to the logits, exactly as in the official code)
            out = out.detach()
            prior = out.softmax(1).mean(0)
            smooth = max(1 / out.shape[0], 1 / out.shape[1]) / torch.max(prior)
            smoothed_prior = (prior + smooth) / (1 + smooth * out.shape[1])
            out = out * smoothed_prior
        self.last_logits = out
        return out.argmax(1)
