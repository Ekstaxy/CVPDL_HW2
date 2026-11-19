"""Sweep for arbn + TENT (entropy minimization on gamma / beta).   .venv/bin/python analysis/sweep_tent.py

Phase A: learning rate / optimizer only (no safeguards) -> how aggressive can we go before it hurts?
Phase B: safeguards at aggressive learning rates: entropy filter, pull-back to source, deep-only, predict-after.
Results -> analysis/sweep_tent.csv
"""
import itertools
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit  # noqa: E402
from tta.arbn_tent import AdaptiveRBNTent  # noqa: E402
from tta.norm import AdaptiveRobustBN  # noqa: E402

rows = []


def go(tag, make, **info):
    t = time.time()
    row = dict(config=tag, **info, **evalkit.evaluate(make))
    row["sweep_seconds"] = time.time() - t
    rows.append(row)
    pd.DataFrame(rows).to_csv(evalkit.ROOT / "analysis/sweep_tent.csv", index=False)
    print(f"{tag:60s} val {row['official']:.4f}  6ord {row['orders_mean']:.4f}/{row['orders_worst']:.4f}  "
          f"long {row['long']:.4f} (1st {row['long_1st_third']:.3f} last {row['long_last_third']:.3f}) "
          f"uniq {row['long_uniq_pred_per_batch']:.1f}", flush=True)


go("arbn (no TENT)", lambda m: AdaptiveRobustBN(m), phase="ref")

# Phase A: lr / optimizer
for opt, lrs in [("sgd", [2.5e-4, 1e-3, 2.5e-3, 5e-3, 1e-2, 2.5e-2, 5e-2, 1e-1]), ("adam", [1e-4, 3e-4, 1e-3, 3e-3])]:
    for lr in lrs:
        go(f"A {opt} lr={lr}", lambda m, lr=lr, opt=opt: AdaptiveRBNTent(m, lr=lr, optim=opt),
           phase="A", optim=opt, lr=lr)

# Phase B: safeguards at aggressive SGD learning rates
for lr, e0, src, layers, after in itertools.product([1e-2, 2.5e-2, 5e-2], [None, 0.4], [None, 0.99],
                                                    ["all", "deep"], [False, True]):
    tag = f"B sgd lr={lr} e0={e0} src={src} layers={layers} after={after}"
    go(tag, lambda m, lr=lr, e0=e0, src=src, layers=layers, after=after:
       AdaptiveRBNTent(m, lr=lr, optim="sgd", e0=e0, src_momentum=src, update_layers=layers, predict_after=after),
       phase="B", optim="sgd", lr=lr, e0=e0, src=src, layers=layers, after=after)
print("saved", evalkit.ROOT / "analysis/sweep_tent.csv")
