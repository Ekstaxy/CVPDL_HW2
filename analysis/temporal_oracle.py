"""Ceiling of temporal-prior approaches: prior_i = TRUE class histogram of the other images within +-w (analysis only)."""
import sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from temporal_sim import load, knn_affinity  # noqa: E402

runs = load()
def run_oracle(w, beta, eps, k, cross_batch):
    accs = []
    for r in runs:
        y = torch.as_tensor(r["y"], device="cuda"); Y = F.one_hot(y, 20).float(); N = len(y)
        preds = []; off = 0
        for raw, t3, ft in r["batches"]:
            n = len(t3); i = torch.arange(off, off + n, device="cuda")
            lo, hi = (0, N) if cross_batch else (off, off + n)
            j = torch.arange(lo, hi, device="cuda")
            K = ((i[:, None] - j[None, :]).abs() <= w) & (i[:, None] != j[None, :])
            pr = K.float() @ Y[lo:hi]; pr = pr / pr.sum(1, keepdim=True).clamp_min(1)
            log_p = t3.log_softmax(1) + beta * torch.log(eps + pr)
            Z = log_p.softmax(1)
            if k:
                W = knn_affinity(ft, k)
                for _ in range(10):
                    Z = (log_p + W @ Z).softmax(1)
            preds.append(Z.argmax(1)); off += n
        accs.append(float((torch.cat(preds) == y).float().mean()))
    return np.mean(accs)
for k in (0, 5):
    for cross in (False, True):
        for w in (2, 4, 8, 16):
            best = max(((run_oracle(w, b, e, k, cross), b, e) for b in (0.5, 1.0, 2.0) for e in (0.01, 0.05)))
            print(f"kNN k={k}  neighbours {'whole stream' if cross else 'same batch only'}  +-{w}: best all_mean {best[0]:.4f} (beta {best[1]}, eps {best[2]})", flush=True)
