"""Method-only GPU time for 192 batches of 64 (the test-stream length), on val images cycled in the long order.
No harness / data loading (those add a constant ~3-4 s). Also times the augmentation alone.

  .venv/bin/python analysis/time_exp.py '<json list of {name, cfg}>' [base config json]
"""
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as E  # noqa: E402
from tta.arbn_teacher import AdaptiveRBNTeacher  # noqa: E402

BASE_CFG = json.load(open(sys.argv[2] if len(sys.argv) > 2 else E.ROOT / "starter/config_h2_fullaug.json"))
IDX = np.concatenate([E.LONG, E.LONG])[: 192 * 64]


def run(name, cfg):
    c = {**BASE_CFG, **cfg}
    if c.get("upscales") is not None:
        c["upscales"] = tuple(c["upscales"])
    torch.manual_seed(0); np.random.seed(0)
    m = AdaptiveRBNTeacher(deepcopy(E.BASE), **c)
    ok, per = 0, []
    torch.cuda.synchronize(); t0 = time.time()
    for s in range(0, len(IDX), 64):
        tb = time.time()
        xb = E.X[IDX[s:s + 64]].cuda().float()
        p = m((xb / 255 - E.MEAN) / E.STD).cpu()
        per.append(time.time() - tb)
        ok += int((p.numpy() == E.y[IDX[s:s + 64]]).sum())
    torch.cuda.synchronize(); t = time.time() - t0
    # augmentation alone (same number of calls as one batch of this config)
    xb = (E.X[:64].cuda().float() / 255 - E.MEAN) / E.STD
    torch.cuda.synchronize(); ta = time.time()
    for _ in range(50):
        m.transform(xb)
    torch.cuda.synchronize(); ta = (time.time() - ta) / 50
    print(f"{name:<46} {t:6.1f} s  ({1000 * np.median(per):5.1f} ms/batch median, max {1000 * np.max(per):5.0f})  "
          f"aug {1000 * ta:5.1f} ms/call  acc {ok / len(IDX):.4f}", flush=True)


if __name__ == "__main__":
    for j in json.loads(sys.argv[1]):
        run(j["name"], j.get("cfg", {}))
