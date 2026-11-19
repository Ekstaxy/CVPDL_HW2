"""Flip-view averaging and feedback of the corrected answer on top of arbn_teacher.
Short streams only (official + 6 orders: every image seen once per run), seeds 0/1/2.
   .venv/bin/python analysis/sweep_feedback.py   -> analysis/sweep_feedback.csv"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as E  # noqa: E402
from tta.arbn_teacher import AdaptiveRBNTeacher  # noqa: E402

CFGS = [dict(), dict(flip=True)]
CFGS += [dict(fb_w=w, fb_mask=m) for m in ("both", "all") for w in (0.25, 0.5, 1.0)]
CFGS += [dict(flip=True, fb_w=0.5, fb_mask="both"), dict(flip=True, fb_w=0.5, fb_mask="all")]


def after_shift(ok, idx):
    d = E.dom[idx]
    starts = [i for i in range(1, len(d)) if d[i] != d[i - 1]]
    return np.mean([ok[s:s + 128].mean() for s in starts])


rows = []
for cfg in CFGS:
    for seed in (0, 1, 2):
        t = time.time()
        r = dict(config=str(cfg) if cfg else "base", seed=seed)
        accs, shifts, fbn = {}, [], []
        for s, idx in E.ORDERS.items():
            pred, m = E.stream(lambda mm: AdaptiveRBNTeacher(mm, **cfg), idx, seed=seed)
            ok = pred == E.y[idx]
            accs[s] = ok.mean(); shifts.append(after_shift(ok, idx)); fbn.append(m.fb_count)
        orders = [v for k, v in accs.items() if k != "official"]
        r.update(official=accs["official"], orders_mean=np.mean(orders), orders_worst=np.min(orders),
                 first128_after_shift=np.mean(shifts), fb_targets_per_run=np.mean(fbn))
        rows.append(r)
        pd.DataFrame(rows).to_csv(E.ROOT / "analysis/sweep_feedback.csv", index=False)
        print(r["config"], seed, {k: round(float(v), 4) for k, v in r.items() if k not in ("config", "seed")},
              f"({time.time() - t:.0f}s)", flush=True)
df = pd.DataFrame(rows)
g = df.groupby("config", sort=False)[["official", "orders_mean", "orders_worst", "first128_after_shift", "fb_targets_per_run"]]
print(pd.concat([g.mean().round(4), g.std()[["official", "orders_mean", "orders_worst"]].round(4).add_suffix("_std")], axis=1).to_string())
