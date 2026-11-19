"""Gain from averaging final probabilities of independent runs on the test-like proxy (upper bound for a
single-pass ensemble of K members)."""
import json, sys
from copy import deepcopy
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import testlike as TL, evalkit as E  # noqa: E402
from tta.arbn_teacher import AdaptiveRBNTeacher  # noqa: E402
X, y, seg = TL.load()
c = json.load(open(sys.argv[1])); c["upscales"] = tuple(c.get("upscales", ()))
P = []
for seed in range(4):
    torch.manual_seed(seed); np.random.seed(seed)
    m = AdaptiveRBNTeacher(deepcopy(E.BASE), **c); ps = []
    for s in range(0, len(X), 64):
        xb = X[s:s + 64].cuda().float(); m((xb / 255 - E.MEAN) / E.STD); ps.append(m.last_final.float().cpu())
    P.append(torch.cat(ps).numpy())
single = [(p.argmax(1) == y).mean() for p in P]
print(f"single {np.mean(single):.4f} {np.round(single, 4)}")
for k in (2, 3, 4):
    print(f"avg of {k}: {(np.mean(P[:k], 0).argmax(1) == y).mean():.4f}")
print("disagree seed0 vs seed1:", (P[0].argmax(1) != P[1].argmax(1)).mean())
