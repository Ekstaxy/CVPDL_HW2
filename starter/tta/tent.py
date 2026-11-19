"""TENT: Fully Test-Time Adaptation by Entropy Minimization (Wang et al., ICLR 2021).

ImageNet-C setting: SGD lr 2.5e-4, momentum 0.9 (nesterov), 1 step per batch, batch-statistics BN,
only BN affine parameters (gamma, beta) are updated. Prediction = output of the same forward pass.
"""
import torch

from .common import bn_affine_params, entropy, force_bn_batch_stats, sgd


class Tent:
    LR = 2.5e-4

    def __init__(self, model):
        self.model = model.eval()
        self.model.requires_grad_(False)
        force_bn_batch_stats(self.model)
        for m in self.model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.requires_grad_(True)
        self.optimizer = sgd(bn_affine_params(self.model), self.LR)
        self.last_logits = None

    @torch.enable_grad()
    def __call__(self, x):
        out = self.model(x)
        loss = entropy(out).mean(0)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.last_logits = out.detach()
        return out.argmax(1)
