"""Paired comparison of arbn_teacher configurations (same seed, same stream) against the first one.

  .venv/bin/python analysis/cfg_paired.py <base config json file> '<json list of {name, cfg}>' [seeds]
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as E  # noqa: E402
from tta.arbn_teacher import AdaptiveRBNTeacher  # noqa: E402

BASE_CFG = json.load(open(sys.argv[1]))
jobs = [{"name": "base", "cfg": {}}] + json.loads(sys.argv[2])
seeds = tuple(int(s) for s in sys.argv[3].split(",")) if len(sys.argv) > 3 else (0, 1, 2)
base = None
for j in jobs:
    c = {**BASE_CFG, **j["cfg"]}; c["upscales"] = tuple(c.get("upscales", ()))
    acc, off, sw, fi, after = [], [], [], [], []
    for seed in seeds:
        for s, idx in E.ORDERS.items():
            pred, _ = E.stream(lambda m: AdaptiveRBNTeacher(m, **c), idx, seed=seed)
            ok = pred == E.y[idx]; d = E.dom[idx]
            acc.append(ok.mean()); sw.append(ok[d != "fisheye"].mean()); fi.append(ok[d == "fisheye"].mean())
            if s == "official":
                off.append(ok.mean())
    a = np.array(acc); base = a if base is None else base
    worst = a.reshape(len(seeds), 7)[:, 1:].min(1).mean()
    print(f"{j['name']:<40} all {a.mean():.4f}  official {np.mean(off):.4f}  6-order worst {worst:.4f}  scan+wave {np.mean(sw):.4f}  "
          f"fisheye {np.mean(fi):.4f}  paired {100 * (a - base).mean():+.2f} (min {100 * (a - base).min():+.2f}, wins {(a > base).sum()}/{len(a)})", flush=True)
