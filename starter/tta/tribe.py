"""TRIBE: Towards Real-World Test-Time Adaptation: Tri-Net Self-Training with Balanced Normalization
(Su, Xu, Jia, AAAI 2024). Port of the official implementation (Gorilla-Lab-SCUT/TRIBE).

  * Balanced BN (BalancedRobustBN2dV5): class-wise running statistics updated with pseudo-labels; the
    normalization statistics are the class-balanced average -> robust to non-i.i.d. / imbalanced batches.
  * Tri-net: student (sees strong augmentation), teacher (shares the student's weights but has its own BN
    statistics, sees clean images, makes predictions), anchor (frozen source weights, own BN statistics)
    regularizing the teacher with an MSE loss.

Hyper-parameters follow the authors' rules of thumb for K = 20 classes:
  ETA = max(0.0005 * K, 0.01) = 0.01, GAMMA = 0.0 (as for CIFAR-10), H0 = 0.05 (as for CIFAR-10, a source model
  with a similar accuracy), LAMBDA = 0.5, Adam with the ImageNet learning rate 5e-4.
Only deviation: a batch where no sample passes the entropy filter is skipped (the reference code would
average over an empty set and produce NaN).
"""
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import NormalizedAug, adam, build_ema, get_tta_transforms, replace_bn, set_submodule


def update_statistics_2d_v5(local_mean, local_var, global_mean, global_var, momentum, data, label, gamma):
    """Verbatim port of `balanced_bn_pyv.update_statistics_2d_v5` (training branch)."""
    l = label.unique()
    l_mapping = torch.arange(l.size(0), device=l.device)
    l_reverse = l.new_full((local_mean.size(0),), -1)
    l_reverse[l] = l_mapping
    label_local = l_reverse[label]
    lm = local_mean[label]
    label_num = torch.zeros_like(l_mapping)
    label_num.scatter_add_(0, label_local, torch.ones_like(label_local))
    mask = label_num > (1.0 / momentum)
    m = torch.where(mask, 1.0 / label_num.float(), torch.full_like(label_num, momentum, dtype=torch.float))

    delta_pre = data - lm[..., None, None]
    delta_k = delta_pre.new_zeros((l.size(0), delta_pre.size(1)))
    delta_k.scatter_add_(0, label_local.unsqueeze(-1).expand(-1, delta_k.size(1)), delta_pre.mean((2, 3)))
    delta_k = delta_k * m.view(-1, 1)
    local_mean[l] = (1.0 - gamma) * delta_k + local_mean[l]
    local_mean[...] = gamma * delta_k.mean(0, keepdim=True) + local_mean

    delta_sq = delta_pre.new_zeros((l.size(0), delta_pre.size(1)))
    delta_sq.scatter_add_(0, label_local.unsqueeze(-1).expand(-1, delta_k.size(1)), delta_pre.pow(2).mean((2, 3)))
    local_var[l] = local_var[l] + (1.0 - gamma) * (m.view(-1, 1) * (delta_sq - label_num.view(-1, 1) * local_var[l])
                                                    - delta_k.pow(2))
    var_gap = (m.view(-1, 1) * delta_sq - delta_k.pow(2)).mean(0, keepdim=True)
    local_var[...] = local_var + gamma * (var_gap - (m * label_num).mean() * local_var)

    global_mean[...] = local_mean.mean(0)
    global_var[...] = local_var.mean(0) + local_mean.var(0)


class BalancedRobustBN2d(nn.Module):
    """Port of `BalancedRobustBN2dV5`."""

    def __init__(self, bn: nn.BatchNorm2d, num_classes, momentum, gamma):
        super().__init__()
        self.num_features, self.num_classes, self.eps = bn.num_features, num_classes, bn.eps
        self.momentum, self.gamma = momentum, gamma
        self.register_buffer("global_mean", deepcopy(bn.running_mean))
        self.register_buffer("global_var", deepcopy(bn.running_var))
        self.weight = deepcopy(bn.weight)
        self.bias = deepcopy(bn.bias)
        self.register_buffer("local_mean", bn.running_mean[None].expand(num_classes, -1).clone())
        self.register_buffer("local_var", bn.running_var[None].expand(num_classes, -1).clone())
        self.label = None

    def forward(self, x):
        self.global_mean = self.global_mean.detach()
        self.global_var = self.global_var.detach()
        self.local_mean = self.local_mean.detach()
        self.local_var = self.local_var.detach()
        if self.label is not None:
            if self.training:
                update_statistics_2d_v5(self.local_mean, self.local_var, self.global_mean, self.global_var,
                                        self.momentum, x, self.label, self.gamma)
            self.label = None
        elif self.training:
            b_var, b_mean = torch.var_mean(x, dim=[0, 2, 3], unbiased=False)
            self.global_mean = (1 - self.momentum) * self.global_mean + self.momentum * b_mean
            self.global_var = (1 - self.momentum) * self.global_var + self.momentum * b_var
        x = (x - self.global_mean[None, :, None, None]) / torch.sqrt(self.global_var[None, :, None, None] + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


class TRIBE:
    LR = 5e-4
    ETA = 0.01
    GAMMA = 0.0
    H0 = 0.05
    LAMBDA = 0.5
    NUM_CLASSES = 20

    def __init__(self, model):
        model.requires_grad_(False)
        model = replace_bn(model, lambda bn: BalancedRobustBN2d(bn, self.NUM_CLASSES, self.ETA, self.GAMMA)).cuda()
        for m in model.modules():
            if isinstance(m, BalancedRobustBN2d):
                m.requires_grad_(True)
        self.model = model
        self.optimizer = adam([p for p in model.parameters() if p.requires_grad], self.LR)
        # teacher: own BN buffers, parameters shared with the student
        self.aux_model = build_ema(model)
        for (name, p_student), _ in zip(self.model.named_parameters(), self.aux_model.named_parameters()):
            set_submodule(self.aux_model, name, p_student)
        # anchor: frozen source weights with its own balanced BN statistics
        self.source_model = build_ema(model)
        self.transform = NormalizedAug(get_tta_transforms(fill=None))
        self.skipped = 0
        self.last_logits = None

    @staticmethod
    def set_bn_label(model, label):
        for m in model.modules():
            if isinstance(m, BalancedRobustBN2d):
                m.label = label

    def __call__(self, x):
        with torch.no_grad():
            self.aux_model.eval()
            ema_out = self.aux_model(x)
        self.update_model(x, ema_out)
        self.last_logits = ema_out
        return ema_out.argmax(1)

    @torch.enable_grad()
    def update_model(self, x, logit):
        p_l = logit.argmax(1)
        self.source_model.train()
        self.aux_model.train()
        self.model.train()
        strong = self.transform(x)

        self.set_bn_label(self.aux_model, p_l)
        ema_sup_out = self.aux_model(x)
        self.set_bn_label(self.model, p_l)
        stu_sup_out = self.model(strong)

        ent = -(ema_sup_out.softmax(-1) * ema_sup_out.log_softmax(-1)).sum(-1)
        mask = ent < self.H0 * math.log(self.NUM_CLASSES)

        with torch.no_grad():
            self.set_bn_label(self.source_model, p_l)
            anchor = self.source_model(x)

        if mask.sum() == 0:
            self.skipped += 1
            return
        l_sup = F.cross_entropy(stu_sup_out, ema_sup_out.argmax(-1), reduction="none")[mask].mean()
        l_reg = self.LAMBDA * F.mse_loss(ema_sup_out, anchor, reduction="none")[mask].mean()
        loss = l_sup + l_reg
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
