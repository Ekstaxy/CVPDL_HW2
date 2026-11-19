"""How much would averaging several independent runs (seeds) of the same online method give? (labelled streams)

Each run is its own single online pass; the final probabilities (after T3A + LAME) are averaged per image.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as E, harsh_val as H  # noqa: E402
from tta.arbn_teacher import AdaptiveRBNTeacher  # noqa: E402
from tta.arbn_tent import lame  # noqa: E402

c = json.load(open(E.ROOT / "starter/config_h2_fast.json")); c["upscales"] = tuple(c["upscales"])
SEEDS = (0, 1, 2, 3, 4, 5)


def probs(X, idx, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    from copy import deepcopy
    m = AdaptiveRBNTeacher(deepcopy(E.BASE), **c); Z = []
    for b in range(0, len(idx), 64):
        x = (X[idx[b:b + 64]].cuda().float() / 255 - E.MEAN) / E.STD
        m(x); lg, ft = m.last_cls
        Z.append(lame(lg, ft, m.lame_k, **m.lame_kw).cpu())
    return torch.cat(Z)


for name, X, idx in (("val official (2,112)", E.X, E.ORDERS["official"]), ("val long (12,672, 18 shifts)", E.X, E.LONG),
                     ("mild proxy (2,112)", None, E.ORDERS["official"])):
    if X is None:
        import os; os.environ["PROXY"] = "mild"
        import importlib; importlib.reload(H); X = H.STREAMS[0][0]
    y = torch.as_tensor(E.y[idx])
    P = [probs(X, idx, s) for s in SEEDS]
    single = [float((p.argmax(1) == y).float().mean()) for p in P]
    line = f"{name:<30} single: mean {np.mean(single):.4f} (min {min(single):.4f}, max {max(single):.4f})"
    for k in (2, 3, 4, 6):
        accs = [float((torch.stack([P[(i + j) % 6] for j in range(k)]).mean(0).argmax(1) == y).float().mean()) for i in range(6)]
        line += f" | avg of {k}: {np.mean(accs):.4f} ({100 * (np.mean(accs) - np.mean(single)):+.2f})"
    print(line, flush=True)
