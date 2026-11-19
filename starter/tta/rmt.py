"""RMT: Robust Mean Teacher for Continual and Gradual Test-Time Adaptation (Döbler et al., CVPR 2023).

ImageNet-C config: SGD lr 0.01 (momentum 0.9, nesterov), teacher EMA 0.999, contrastive temperature 0.1,
projection dim 128, contrast mode "all", lambda_ce_trg = lambda_cont = 1. All parameters trained, BN uses
test-batch statistics. Prediction = student + teacher logits.

Deviations forced by the HW2 rules (no source data allowed):
  * no source warm-up (NUM_SAMPLES_WARM_UP = 0) and no source replay (LAMBDA_CE_SRC = 0, the "source-free" variant);
  * source class prototypes (mean source features) are replaced by the classifier weight vectors fc.weight[c],
    the only class-wise source information available in the checkpoint.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import NormalizedAug, build_ema, ema_update, force_bn_batch_stats, get_tta_transforms, sgd, \
    symmetric_cross_entropy


class RMT:
    LR = 0.01
    MT = 0.999
    TEMPERATURE = 0.1
    PROJECTION_DIM = 128
    LAMBDA_CE_TRG = 1.0
    LAMBDA_CONT = 1.0

    def __init__(self, model):
        self.model = model.eval()
        force_bn_batch_stats(self.model)
        self.model.requires_grad_(True)
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1], nn.Flatten())
        self.classifier = self.model.fc
        self.optimizer = sgd(self.model.parameters(), self.LR)
        self.model_ema = build_ema(self.model)
        self.transform = NormalizedAug(get_tta_transforms())
        # stand-in for source prototypes: classifier weights (K, D)
        self.prototypes = self.model.fc.weight.detach().clone()
        d = self.prototypes.shape[1]
        self.projector = nn.Sequential(nn.Linear(d, self.PROJECTION_DIM), nn.ReLU(),
                                       nn.Linear(self.PROJECTION_DIM, self.PROJECTION_DIM)).cuda()
        self.optimizer.add_param_group({"params": self.projector.parameters(), "lr": self.LR})
        self.last_logits = None

    def contrastive_loss(self, features):
        """SupCon/SimCLR loss with labels=None, mode 'all' (views = [prototype, test, augmented test])."""
        b, n_views = features.shape[:2]
        mask = torch.eye(b, device=features.device)
        contrast = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast = F.normalize(self.projector(contrast), p=2, dim=1)
        logits = contrast @ contrast.T / self.TEMPERATURE
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
        mask = mask.repeat(n_views, n_views)
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        return -mean_log_prob_pos.view(n_views, b).mean()

    @torch.enable_grad()
    def __call__(self, x):
        f = self.feature_extractor(x)
        out = self.classifier(f)
        f_aug = self.feature_extractor(self.transform(x))
        out_aug = self.classifier(f_aug)
        with torch.no_grad():
            out_ema = self.model_ema(x)
            sim = F.normalize(f, dim=1) @ F.normalize(self.prototypes, dim=1).T  # (B, K)
            nearest = sim.argmax(1)
        feats = torch.stack([self.prototypes[nearest], f, f_aug], dim=1)
        loss_cont = self.contrastive_loss(feats)
        loss_st = (0.5 * symmetric_cross_entropy(out, out_ema) + 0.5 * symmetric_cross_entropy(out_aug, out_ema)).mean(0)
        loss = self.LAMBDA_CE_TRG * loss_st + self.LAMBDA_CONT * loss_cont
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        ema_update(self.model_ema, self.model, self.MT)
        logits = (out + out_ema).detach()
        self.last_logits = logits
        return logits.argmax(1)
