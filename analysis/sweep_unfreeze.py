"""Unfreeze the last residual block / stage on top of arbn_teacher (mt 0.999, w_ent 1, lr 0.1, student + LAME).
   .venv/bin/python analysis/sweep_unfreeze.py   -> analysis/sweep_unfreeze.csv"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as E  # noqa: E402
from tta.arbn_teacher import AdaptiveRBNTeacher  # noqa: E402

streams = dict(E.ORDERS); streams["long"] = E.LONG
rows = []
cfgs = [dict(unfreeze=None)] + [dict(unfreeze=u, lr_conv=lr) for u in ("layer4.2", "layer4")
                                for lr in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)]
for cfg in cfgs:
    t = time.time()
    acc = {s: (E.stream(lambda m: AdaptiveRBNTeacher(m, **cfg), idx)[0] == E.y[idx]).mean() for s, idx in streams.items()}
    orders = [v for k, v in acc.items() if k not in ("official", "long")]
    rows.append(dict(**{k: str(v) for k, v in cfg.items()}, official=acc["official"], orders_mean=np.mean(orders),
                     orders_worst=np.min(orders), long=acc["long"]))
    pd.DataFrame(rows).to_csv(E.ROOT / "analysis/sweep_unfreeze.csv", index=False)
    r = rows[-1]
    print(f"{str(cfg):45s} ({time.time() - t:.0f}s) val {r['official']:.4f} 6ord {r['orders_mean']:.4f}/{r['orders_worst']:.4f} long {r['long']:.4f}", flush=True)
