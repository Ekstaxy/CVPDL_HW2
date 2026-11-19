"""T3A: Test-Time Classifier Adjustment Module (Iwasawa & Matsuo, NeurIPS 2021), output-only.

Per class, keep the FILTER_K lowest-entropy test features pseudo-labelled by the linear classifier (the classifier
weight vectors seed the support set), and classify by  z @ normalize(sum of normalized supports of each class).
Supports that drop out of a class's top-K can never come back (new supports only add competition), so pruning to
the selected set after every batch is exact and keeps memory bounded.
"""
import torch
import torch.nn.functional as F


class T3A:
    def __init__(self, fc: torch.nn.Linear, filter_k=50, num_classes=20):
        self.k, self.c = filter_k, num_classes
        w = fc.weight.detach().float().clone()
        p0 = (w @ w.T + fc.bias.detach().float()).softmax(1)
        self.s = w
        self.y = p0.argmax(1)
        self.e = -(p0 * p0.clamp_min(1e-12).log()).sum(1)

    @torch.no_grad()
    def __call__(self, z, logits):
        z = z.float()
        p = logits.float().softmax(1)
        S = torch.cat([self.s, z])
        Y = torch.cat([self.y, p.argmax(1)])
        Ent = torch.cat([self.e, -(p * p.clamp_min(1e-12).log()).sum(1)])
        keep = torch.zeros(len(S), dtype=torch.bool, device=S.device)
        for c in range(self.c):
            ic = (Y == c).nonzero().flatten()
            if len(ic):
                keep[ic[Ent[ic].argsort()[: self.k]]] = True
        self.s, self.y, self.e = S[keep], Y[keep], Ent[keep]
        proto = F.normalize(F.one_hot(self.y, self.c).float().T @ F.normalize(self.s, dim=1), dim=1)
        return z @ proto.T

    @torch.no_grad()
    def peek(self, z):
        """Scores from the current support set only (nothing is added)."""
        proto = F.normalize(F.one_hot(self.y, self.c).float().T @ F.normalize(self.s, dim=1), dim=1)
        return z.float() @ proto.T
