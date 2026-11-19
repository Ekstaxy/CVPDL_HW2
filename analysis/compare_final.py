"""Seed-averaged comparison of the candidate methods.   .venv/bin/python analysis/compare_final.py [extra]
Deterministic methods are run once; stochastic ones (strong augmentation) with seeds 0, 1, 2."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as E  # noqa: E402
from tta.arbn_teacher import AdaptiveRBNTeacher  # noqa: E402
from tta.arbn_tent import AdaptiveRBNTent  # noqa: E402

streams = dict(E.ORDERS); streams["long"] = E.LONG
CANDS = {
    "arbn_tent + LAME": (lambda m: AdaptiveRBNTent(m), [0]),
    "arbn_tent + T3A + LAME": (lambda m: AdaptiveRBNTent(m, t3a_k=50), [0]),
    "arbn_teacher + LAME": (lambda m: AdaptiveRBNTeacher(m), [0, 1, 2]),
    "arbn_teacher + T3A + LAME": (lambda m: AdaptiveRBNTeacher(m, t3a_k=50), [0, 1, 2]),
}
rows = []
for name, (make, seeds) in CANDS.items():
    for seed in seeds:
        t = time.time()
        acc = {s: (E.stream(make, idx, seed=seed)[0] == E.y[idx]).mean() for s, idx in streams.items()}
        orders = [v for k, v in acc.items() if k not in ("official", "long")]
        rows.append(dict(method=name, seed=seed, official=acc["official"], orders_mean=np.mean(orders),
                         orders_worst=np.min(orders), long=acc["long"]))
        print(name, seed, {k: round(v, 4) for k, v in rows[-1].items() if k not in ("method", "seed")},
              f"({time.time() - t:.0f}s)", flush=True)
df = pd.DataFrame(rows); df.to_csv(E.ROOT / "analysis/compare_final.csv", index=False)
g = df.groupby("method", sort=False)[["official", "orders_mean", "orders_worst", "long"]]
print(pd.concat([g.mean().round(4), g.std().round(4).add_suffix("_std")], axis=1).to_string())
