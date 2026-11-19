"""Real paired check of the LAME graph options: one run of arbn_best per (seed, stream); the final stage is replayed
on the recorded T3A logits / features (the final answer never feeds back into the adaptation, so this is exact).

  .venv/bin/python analysis/lame_paired.py
"""
import json, sys
from copy import deepcopy
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as E  # noqa: E402
from tta.arbn_teacher import AdaptiveRBNTeacher  # noqa: E402
from tta.arbn_tent import lame  # noqa: E402

CFG = json.load(open(E.ROOT / "starter/best_config.json")); CFG["upscales"] = tuple(CFG["upscales"])
VAR = {"base (arbn_best)": {}}
for pw in (False, True):
    for tw, tau in ((0.0, 4), (0.1, 4), (0.15, 4), (0.25, 4), (0.1, 8), (0.15, 6)):
        if pw or tw:
            VAR[f"pow={int(pw)} time {tw}*box({tau})"] = dict(power=pw, time_w=tw, time_tau=tau)
res = {v: dict(acc=[], off=[], sw=[], fi=[]) for v in VAR}
for seed in (0, 1, 2):
    for s, idx in E.ORDERS.items():
        torch.manual_seed(seed); np.random.seed(seed)
        m = AdaptiveRBNTeacher(deepcopy(E.BASE), **CFG)
        P = {v: [] for v in VAR}
        for b in range(0, len(idx), 64):
            xb = E.X[idx[b:b + 64]].cuda().float()
            out = m((xb / 255 - E.MEAN) / E.STD)
            lg, ft = m.last_cls
            for v, kw in VAR.items():
                P[v].append(lame(lg, ft, 5, **kw).argmax(1).cpu())
            assert (P["base (arbn_best)"][-1] == out.cpu()).all()
        d = E.dom[idx]
        for v in VAR:
            ok = torch.cat(P[v]).numpy() == E.y[idx]
            res[v]["acc"].append(ok.mean()); res[v]["sw"].append(ok[d != "fisheye"].mean()); res[v]["fi"].append(ok[d == "fisheye"].mean())
            if s == "official":
                res[v]["off"].append(ok.mean())
    print("seed", seed, "done", flush=True)
b = np.array(res["base (arbn_best)"]["acc"])
for v, r in res.items():
    a = np.array(r["acc"]); o = np.array(a).reshape(3, 7)[:, 1:]
    print(f"{v:<28} all {a.mean():.4f}  official {np.mean(r['off']):.4f} ({' '.join(f'{x:.4f}' for x in r['off'])})  6-order worst {o.min(1).mean():.4f}  "
          f"scan+wave {np.mean(r['sw']):.4f}  fisheye {np.mean(r['fi']):.4f}  paired {100 * (a - b).mean():+.2f} (min {100 * (a - b).min():+.2f}, wins {(a > b).sum()}/21)")
