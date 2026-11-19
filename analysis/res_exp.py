"""Run the whole arbn_best pipeline at a different base resolution (the batch is resized before the method sees it).

  .venv/bin/python analysis/res_exp.py '<json list of {name, res, mode, cfg}>' [seeds]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as E  # noqa: E402
from tta.arbn_teacher import AdaptiveRBNTeacher  # noqa: E402

CFG = json.load(open(E.ROOT / "starter/best_config.json")); CFG["upscales"] = tuple(CFG["upscales"])


class Resized:
    def __init__(self, model, res, mode, **cfg):
        self.m = AdaptiveRBNTeacher(model, **cfg); self.res, self.mode = res, mode

    def __call__(self, x):
        if self.res != x.shape[-1]:
            kw = dict(align_corners=False) if self.mode != "nearest" else {}
            x = F.interpolate(x, size=(self.res, self.res), mode=self.mode, **kw)
        return self.m(x)


def run(name, res=160, mode="bilinear", cfg=None, seeds=(0,)):
    c = {**CFG, **(cfg or {})}
    if "upscales" in c:
        c["upscales"] = tuple(c["upscales"])
    make = lambda m: Resized(m, res, mode, **c)
    accs, off, sw, fi = [], [], [], []
    for seed in seeds:
        for s, idx in E.ORDERS.items():
            pred, _ = E.stream(make, idx, seed=seed)
            ok = pred == E.y[idx]; d = E.dom[idx]
            accs.append(ok.mean()); sw.append(ok[d != "fisheye"].mean()); fi.append(ok[d == "fisheye"].mean())
            if s == "official":
                off.append(ok.mean())
    print(f"{name:<44} all {np.mean(accs):.4f}  official {np.mean(off):.4f}  worst {np.min(accs):.4f}  "
          f"scan+wave {np.mean(sw):.4f}  fisheye {np.mean(fi):.4f}", flush=True)
    return np.array(accs)


if __name__ == "__main__":
    jobs = json.loads(sys.argv[1])
    seeds = tuple(int(s) for s in sys.argv[2].split(",")) if len(sys.argv) > 2 else (0,)
    for j in jobs:
        run(j["name"], j.get("res", 160), j.get("mode", "bilinear"), j.get("cfg"), seeds)
