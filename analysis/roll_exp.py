"""Is the weak start of every domain a position effect (adaptation lag) or batch content?

Each domain is cut to whole batches and streamed with circular shifts (roll r batches), either from the source model
or right after another full domain. Every image is then seen at several stream positions; accuracy by position,
averaged over rolls, removes the content effect.

  .venv/bin/python analysis/roll_exp.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as E  # noqa: E402
from tta.arbn_teacher import AdaptiveRBNTeacher  # noqa: E402

CFG = json.load(open(E.ROOT / "starter/best_config.json")); CFG["upscales"] = tuple(CFG["upscales"])
EXTRA = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
make = lambda m: AdaptiveRBNTeacher(m, **{**CFG, **EXTRA})
NB = 9  # batches kept per domain (576 images: the shortest domain has 612)
ROLLS = (0, 3, 6)

pos_ok = {"source": np.zeros((NB, 0)).tolist(), "after": np.zeros((NB, 0)).tolist()}
acc = {"source": [[] for _ in range(NB)], "after": [[] for _ in range(NB)]}
for seed in (0, 1, 2):
    for d in E.DOMS:
        idx_d = np.where(E.dom == d)[0][: NB * 64]
        for r in ROLLS:
            idx = np.roll(idx_d, -64 * r)
            for mode in ("source", "after"):
                if mode == "source":
                    full = idx
                else:
                    other = E.DOMS[(E.DOMS.index(d) + 1) % 3]
                    pre = np.where(E.dom == other)[0][: NB * 64]
                    full = np.concatenate([pre, idx])
                pred, _ = E.stream(make, full, seed=seed)
                ok = (pred == E.y[full])[-NB * 64:]
                for b in range(NB):
                    acc[mode][b].append(ok[b * 64:(b + 1) * 64].mean())
print("config extra:", EXTRA)
for mode in ("source", "after"):
    a = np.array([np.mean(v) for v in acc[mode]])
    print(f"{mode:7s} accuracy by batch position in the domain: " + " ".join(f"{v:.3f}" for v in a)
          + f" | mean {a.mean():.4f} | steady (pos>=3) {a[3:].mean():.4f} | lag cost per domain {64 * (a[3:].mean() - a[:3]).sum():.1f} images")
