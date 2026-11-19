"""Hyper-parameter sweep for Robust BN (fixed alpha) and Adaptive Robust BN.

  .venv/bin/python analysis/sweep_arbn.py

Evaluation (offline, val only):
  * "official": the val stream in its official order (identical to harness run_stream, verified below);
  * "6 orders": the three val domains concatenated in all 6 orders (within-domain order kept, batches of 64
    straddle the boundaries) -> less overfitting to one particular shift sequence. Reports mean and worst.
Results -> analysis/sweep_arbn.csv
"""
import itertools
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
from run import load_model  # noqa: E402
from tta.norm import AdaptiveRobustBN, RobustBN  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
DOMS = ["scanline", "wave", "fisheye"]

ids = pd.read_csv(ROOT / "data/val/stream.csv")["id"].tolist()
y = pd.read_csv(ROOT / "data/val/labels.csv").set_index("id").loc[ids, "label"].to_numpy()
dom = pd.read_csv(ROOT / "analysis/val_domains.csv").set_index("id").loc[ids, "domain"].to_numpy()
X = torch.from_numpy(np.stack([np.asarray(Image.open(ROOT / f"data/val/images/{i}.png").convert("RGB")) for i in ids]))
X = X.permute(0, 3, 1, 2).contiguous()

orders = {"official": np.arange(len(ids))}
for perm in itertools.permutations(DOMS):
    orders["-".join(d[:4] for d in perm)] = np.concatenate([np.where(dom == d)[0] for d in perm])

BASE = load_model(str(ROOT / "source_resnet50_160.pth"))


def evaluate(make, idx):
    method = make(deepcopy(BASE))
    preds = []
    for s in range(0, len(idx), 64):
        xb = X[idx[s:s + 64]].cuda().float()
        preds.append(method((xb / 255 - MEAN) / STD).cpu())
    return (torch.cat(preds).numpy() == y[idx]), method


def run_config(tag, make):
    row = {"config": tag}
    accs = []
    for name, idx in orders.items():
        ok, method = evaluate(make, idx)
        if name == "official":
            row["official"] = ok.mean()
            for d in DOMS:
                row[d] = ok[dom == d].mean()
            starts = [0, 777, 1500]
            row["first128_after_shift"] = np.mean([ok[s:s + 128].mean() for s in starts[1:]])
            row["mean_alpha"] = np.mean(getattr(method, "alpha_hist", [np.nan]) or [np.nan])
        else:
            accs.append(ok.mean())
    row["orders_mean"], row["orders_worst"] = np.mean(accs), np.min(accs)
    return row


rows = []
with torch.no_grad():
    for a in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        R = type("R", (RobustBN,), {"ALPHA": a})
        rows.append(dict(run_config(f"rbn a={a}", R), method="rbn", alpha=a))
        print(rows[-1]["config"], round(rows[-1]["official"], 4), round(rows[-1]["orders_mean"], 4), flush=True)
    for det, amin, amax, g in itertools.product(["shallow", "per_layer"], [0.02, 0.05, 0.1], [0.3, 0.5, 1.0],
                                                [0.1, 0.25, 0.5, 1.0, 2.0]):
        tag = f"arbn {det} min={amin} max={amax} gain={g}"
        make = lambda m, det=det, amin=amin, amax=amax, g=g: AdaptiveRobustBN(m, amin, amax, g, det)
        rows.append(dict(run_config(tag, make), method="arbn", detect=det, alpha_min=amin, alpha_max=amax, gain=g))
        print(tag, round(rows[-1]["official"], 4), round(rows[-1]["orders_mean"], 4), flush=True)

df = pd.DataFrame(rows)
df.to_csv(ROOT / "analysis/sweep_arbn.csv", index=False)
print("saved", ROOT / "analysis/sweep_arbn.csv")
