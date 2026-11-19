"""Offline val evaluation helpers (analysis only; methods never see this).

  official : the val stream in its official order (same result as the harness)
  orders   : the 3 val domains concatenated in all 6 orders (within-domain order kept)
  long     : the 6 orders back to back = 12,672 images / 198 batches (about the test-stream length,
             18 domain shifts) -> exposes slow degradation / collapse that the 33-batch val cannot show
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

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
DOMS = ["scanline", "wave", "fisheye"]

ids = pd.read_csv(ROOT / "data/val/stream.csv")["id"].tolist()
y = pd.read_csv(ROOT / "data/val/labels.csv").set_index("id").loc[ids, "label"].to_numpy()
dom = pd.read_csv(ROOT / "analysis/val_domains.csv").set_index("id").loc[ids, "domain"].to_numpy()
X = torch.from_numpy(np.stack([np.asarray(Image.open(ROOT / f"data/val/images/{i}.png").convert("RGB")) for i in ids]))
X = X.permute(0, 3, 1, 2).contiguous()

ORDERS = {"official": np.arange(len(ids))}
for perm in itertools.permutations(DOMS):
    ORDERS["-".join(d[:4] for d in perm)] = np.concatenate([np.where(dom == d)[0] for d in perm])
LONG = np.concatenate([v for k, v in ORDERS.items() if k != "official"])

BASE = load_model(str(ROOT / "source_resnet50_160.pth"))


def stream(make, idx, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    method = make(deepcopy(BASE))
    preds = []
    for s in range(0, len(idx), 64):
        xb = X[idx[s:s + 64]].cuda().float()
        preds.append(method((xb / 255 - MEAN) / STD).cpu())
    pred = torch.cat(preds).numpy()
    return pred, method


def evaluate(make, do_orders=True, do_long=True):
    row = {}
    pred, m = stream(make, ORDERS["official"])
    ok = pred == y
    row["official"] = ok.mean()
    for d in DOMS:
        row[d] = ok[dom == d].mean()
    row["first128_after_shift"] = np.mean([ok[s:s + 128].mean() for s in (777, 1500)])
    if do_orders:
        accs = [(stream(make, idx)[0] == y[idx]).mean() for k, idx in ORDERS.items() if k != "official"]
        row["orders_mean"], row["orders_worst"] = np.mean(accs), np.min(accs)
    if do_long:
        pred, m = stream(make, LONG)
        ok = pred == y[LONG]
        n = len(ok) // 3
        row["long"] = ok.mean()
        row["long_1st_third"], row["long_last_third"] = ok[:n].mean(), ok[-n:].mean()
        nb = len(pred) // 64
        row["long_uniq_pred_per_batch"] = np.mean([len(np.unique(pred[b * 64:(b + 1) * 64])) for b in range(nb)])
        row["skipped"] = getattr(m, "skipped", np.nan)
    return row
