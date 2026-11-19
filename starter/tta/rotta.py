"""RoTTA: Robust Test-Time Adaptation in Dynamic Scenarios (Yuan et al., CVPR 2023).

Port of the official implementation (BIT-DA/RoTTA). Default config: Adam lr 1e-3, alpha (RBN) 0.05,
nu (teacher EMA) 0.001, memory size 64, update every 64 samples, lambda_t = lambda_u = 1.

  * Robust BN with EMA statistics (replaces every BN layer).
  * CSTU memory: category-balanced sampling with timeliness and uncertainty.
  * Teacher (EMA) predicts; student is trained on the strongly augmented memory with the teacher's soft labels,
    with each sample weighted by its age (time-aware reweighting).
"""
import math

import torch

from .common import NormalizedAug, adam, build_ema, ema_update, get_tta_transforms, replace_bn, softmax_entropy
from .norm import RobustBN2d


class MemoryItem:
    def __init__(self, data, uncertainty, age=0):
        self.data, self.uncertainty, self.age = data, uncertainty, age


class CSTU:
    """Category-balanced Sampling with Timeliness and Uncertainty (port of `core/utils/memory.CSTU`)."""

    def __init__(self, capacity, num_class, lambda_t=1.0, lambda_u=1.0):
        self.capacity, self.num_class = capacity, num_class
        self.per_class = capacity / num_class
        self.lambda_t, self.lambda_u = lambda_t, lambda_u
        self.data = [[] for _ in range(num_class)]

    def occupancy(self):
        return sum(len(c) for c in self.data)

    def add_instance(self, x, prediction, uncertainty):
        new_score = self.heuristic_score(0, uncertainty)
        if self.remove_instance(prediction, new_score):
            self.data[prediction].append(MemoryItem(x, uncertainty, 0))
        for c in self.data:
            for item in c:
                item.age += 1

    def remove_instance(self, cls, score):
        if len(self.data[cls]) < self.per_class:
            if self.occupancy() < self.capacity:
                return True
            counts = [len(c) for c in self.data]
            majority = [i for i, n in enumerate(counts) if n == max(counts)]
            return self.remove_from_classes(majority, score)
        return self.remove_from_classes([cls], score)

    def remove_from_classes(self, classes, score_base):
        max_class = max_index = max_score = None
        for cls in classes:
            for idx, item in enumerate(self.data[cls]):
                s = self.heuristic_score(item.age, item.uncertainty)
                if max_score is None or s >= max_score:
                    max_score, max_index, max_class = s, idx, cls
        if max_class is None:
            return True
        if max_score > score_base:
            self.data[max_class].pop(max_index)
            return True
        return False

    def heuristic_score(self, age, uncertainty):
        return self.lambda_t * 1 / (1 + math.exp(-age / self.capacity)) + \
            self.lambda_u * uncertainty / math.log(self.num_class)

    def get_memory(self):
        data, ages = [], []
        for c in self.data:
            for item in c:
                data.append(item.data)
                ages.append(item.age / self.capacity)
        return data, ages


def timeliness_reweighting(ages, device):
    ages = torch.tensor(ages, dtype=torch.float32, device=device)
    return torch.exp(-ages) / (1 + torch.exp(-ages))


class RoTTA:
    LR = 1e-3
    ALPHA = 0.05
    NU = 0.001
    MEMORY_SIZE = 64
    UPDATE_FREQUENCY = 64
    LAMBDA_T = 1.0
    LAMBDA_U = 1.0
    NUM_CLASSES = 20

    def __init__(self, model):
        model.requires_grad_(False)
        model = replace_bn(model, lambda bn: RobustBN2d(bn, self.ALPHA)).cuda()
        for m in model.modules():
            if isinstance(m, RobustBN2d):
                m.requires_grad_(True)
        self.model = model
        self.optimizer = adam([p for p in model.parameters() if p.requires_grad], self.LR)
        self.model_ema = build_ema(model)
        self.mem = CSTU(self.MEMORY_SIZE, self.NUM_CLASSES, self.LAMBDA_T, self.LAMBDA_U)
        self.transform = NormalizedAug(get_tta_transforms(fill=None))
        self.current_instance = 0
        self.last_logits = None

    def __call__(self, x):
        with torch.no_grad():
            self.model.eval()
            self.model_ema.eval()
            ema_out = self.model_ema(x)
            prob = ema_out.softmax(1)
            pseudo = prob.argmax(1).tolist()
            unc = (-(prob * torch.log(prob + 1e-6)).sum(1)).tolist()
        for i in range(x.shape[0]):
            self.mem.add_instance(x[i], pseudo[i], unc[i])
            self.current_instance += 1
            if self.current_instance % self.UPDATE_FREQUENCY == 0:
                self.update_model()
        self.last_logits = ema_out
        return ema_out.argmax(1)

    @torch.enable_grad()
    def update_model(self):
        self.model.train()
        self.model_ema.train()
        sup_data, ages = self.mem.get_memory()
        if len(sup_data) > 0:
            sup_data = torch.stack(sup_data)
            strong = self.transform(sup_data)
            with torch.no_grad():
                ema_sup_out = self.model_ema(sup_data)
            stu_sup_out = self.model(strong)
            w = timeliness_reweighting(ages, sup_data.device)
            loss = (softmax_entropy(stu_sup_out, ema_sup_out) * w).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        ema_update(self.model_ema, self.model, 1 - self.NU)
