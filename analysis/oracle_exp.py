"""Upper bound (analysis only, uses val labels -> never a method): replace the entropy loss by supervised
cross-entropy with the true labels. Tells whether BN gamma/beta capacity or pseudo-label quality is the limit.

  .venv/bin/python analysis/oracle_exp.py
"""
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as E  # noqa: E402
import tta.arbn_teacher as AT  # noqa: E402

CFG = json.load(open(E.ROOT / "starter/best_config.json")); CFG["upscales"] = tuple(CFG["upscales"])
ORIG_ENT = AT.entropy
CUR = {}


def run(name, oracle, noise=0.0, **extra):
    AT.entropy = (lambda out: F.cross_entropy(out, CUR["y"], reduction="none")) if oracle else ORIG_ENT
    res = {"final": [], "raw": [], "sw": [], "fi": []}
    for s, idx in E.ORDERS.items():
        torch.manual_seed(0); np.random.seed(0)
        g = torch.Generator().manual_seed(1)
        m = AT.AdaptiveRBNTeacher(deepcopy(E.BASE), **{**CFG, **extra})
        fin, raw = [], []
        for b in range(0, len(idx), 64):
            xb = X = E.X[idx[b:b + 64]].cuda().float()
            yb = torch.from_numpy(E.y[idx[b:b + 64]]).cuda()
            if noise > 0:  # wrong labels on a random fraction (to mimic imperfect pseudo-labels)
                flip = torch.rand(len(yb), generator=g).cuda() < noise
                yb = torch.where(flip, torch.randint(0, 20, (len(yb),), generator=g).cuda(), yb)
            CUR["y"] = yb
            fin.append(m((xb / 255 - E.MEAN) / E.STD).cpu())
            raw.append(m.last_outputs["student"][0].argmax(1).cpu())
        fin, raw = torch.cat(fin).numpy(), torch.cat(raw).numpy()
        ok = fin == E.y[idx]; d = E.dom[idx]
        res["final"].append(ok.mean()); res["raw"].append((raw == E.y[idx]).mean())
        res["sw"].append(ok[d != "fisheye"].mean()); res["fi"].append(ok[d == "fisheye"].mean())
    print(f"{name:<52} final {np.mean(res['final']):.4f} (official {res['final'][0]:.4f})  raw view-avg {np.mean(res['raw']):.4f}  "
          f"scan+wave {np.mean(res['sw']):.4f}  fisheye {np.mean(res['fi']):.4f}", flush=True)
    AT.entropy = ORIG_ENT


if __name__ == "__main__":
    run("baseline (entropy + consistency)", False)
    run("ORACLE CE on gamma/beta + consistency", True)
    run("ORACLE CE on gamma/beta, no consistency", True, w_cons=0.0)
    run("ORACLE CE, 10% wrong labels + consistency", True, noise=0.10)
    run("ORACLE CE + layer4 unfrozen (lr_conv 1e-3)", True, unfreeze="layer4")
