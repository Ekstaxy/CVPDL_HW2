"""Sweep arbn + TENT + mean teacher (BN gamma/beta only).   .venv/bin/python analysis/sweep_teacher.py

One run per (MT, W_ENT, LR) scores all three prediction sources (student / teacher / ensemble) with and
without LAME, on official val, the 6 domain orders and the long stream. Results -> analysis/sweep_teacher.csv
"""
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as E  # noqa: E402
from tta.arbn_teacher import AdaptiveRBNTeacher  # noqa: E402
from tta.arbn_tent import lame  # noqa: E402

KEYS = [(p, l) for p in ("student", "teacher", "ensemble") for l in (False, True)]
streams = dict(E.ORDERS); streams["long"] = E.LONG


def run(cfg, idx):
    outs = {k: [] for k in KEYS}

    class R(AdaptiveRBNTeacher):
        def __call__(s, x):
            p = super().__call__(x)
            for (src, use_lame) in KEYS:
                lg, f = s.last_outputs[src]
                outs[(src, use_lame)].append((lame(lg, f, 5) if use_lame else lg).argmax(1).cpu())
            return p

    E.stream(lambda m: R(m, **cfg), idx)
    return {k: torch.cat(v).numpy() for k, v in outs.items()}


rows = []
for mt, w_ent, lr in itertools.product([0.9, 0.99, 0.999], [1.0, 0.0], [0.1, 0.05]):
    cfg = dict(mt=mt, w_ent=w_ent, lr=lr)
    t = time.time()
    res = {s: run(cfg, idx) for s, idx in streams.items()}
    for (src, use_lame) in KEYS:
        acc = {s: (res[s][(src, use_lame)] == E.y[idx]).mean() for s, idx in streams.items()}
        orders = [v for k, v in acc.items() if k not in ("official", "long")]
        rows.append(dict(mt=mt, w_ent=w_ent, lr=lr, pred=src, lame=use_lame, official=acc["official"],
                         orders_mean=np.mean(orders), orders_worst=np.min(orders), long=acc["long"]))
    pd.DataFrame(rows).to_csv(E.ROOT / "analysis/sweep_teacher.csv", index=False)
    best = max(rows[-6:], key=lambda r: r["orders_mean"])
    print(f"mt={mt} w_ent={w_ent} lr={lr} ({time.time() - t:.0f}s)  best: pred={best['pred']} lame={best['lame']} "
          f"val {best['official']:.4f} 6ord {best['orders_mean']:.4f}/{best['orders_worst']:.4f} long {best['long']:.4f}",
          flush=True)
print("saved")
