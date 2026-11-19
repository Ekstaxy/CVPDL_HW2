"""Generic paired runner: 3 seeds x 7 val streams per configuration, resumable, summary table vs a reference.

  .venv/bin/python analysis/ablation.py <jobs.json> <out prefix> [reference name]
jobs.json: list of {"name", "method" (key of methods.METHODS, default arbn_teacher), "cfg" (kwargs)}
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as E  # noqa: E402
from methods import METHODS  # noqa: E402

jobs = json.load(open(sys.argv[1])); out = Path(sys.argv[2]); ref = sys.argv[3] if len(sys.argv) > 3 else jobs[0]["name"]
csv = out.with_suffix(".csv")
rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
done = {(r["name"], r["seed"], r["stream"]) for r in rows}
for j in jobs:
    cfg = dict(j.get("cfg", {}))
    if "upscales" in cfg: cfg["upscales"] = tuple(cfg["upscales"])
    cls = METHODS[j.get("method", "arbn_teacher")]
    det = j.get("method", "arbn_teacher") in ("source", "bn_adapt", "rbn", "arbn", "arbn_lame")
    for seed in (0, 1, 2):
        for s, idx in E.ORDERS.items():
            if (j["name"], seed, s) in done: continue
            if det and seed > 0:   # deterministic methods: copy seed 0
                r0 = next(r for r in rows if r["name"] == j["name"] and r["seed"] == 0 and r["stream"] == s)
                rows.append({**r0, "seed": seed}); continue
            pred, _ = E.stream(lambda m: cls(m, **cfg), idx, seed=seed)
            ok = pred == E.y[idx]; d = E.dom[idx]
            b = [int(np.where(d[1:] != d[:-1])[0][k] + 1) for k in range(2)]
            rows.append(dict(name=j["name"], seed=seed, stream=s, acc=ok.mean(),
                             after=np.mean([ok[x:x + 128].mean() for x in b]),
                             **{f"c_{k}": int(ok[d == k].sum()) for k in E.DOMS}, **{f"n_{k}": int((d == k).sum()) for k in E.DOMS}))
        pd.DataFrame(rows).to_csv(csv, index=False)
    print("done", j["name"], flush=True)
R = pd.DataFrame(rows)
base = R[R.name == ref].sort_values(["seed", "stream"]).acc.to_numpy()
lines = ["| Config | official val | 7-stream mean | 6-order worst | scanline | wave | fisheye | after shift | paired vs ref |", "|---|---|---|---|---|---|---|---|---|"]
for j in jobs:
    g = R[R.name == j["name"]].sort_values(["seed", "stream"]); a = g.acc.to_numpy()
    worst = g[g.stream != "official"].groupby("seed").acc.min().mean()
    dom = [g[f"c_{k}"].sum() / g[f"n_{k}"].sum() for k in E.DOMS]
    d = 100 * (a - base)
    lines.append(f"| {j['name']} | {g[g.stream == 'official'].acc.mean():.4f} | {a.mean():.4f} | {worst:.4f} | {dom[0]:.3f} | {dom[1]:.3f} | {dom[2]:.3f} | "
                 f"{g.after.mean():.3f} | {d.mean():+.2f} ({(d > 0).sum()}/21, min {d.min():+.2f}) |")
out.with_suffix(".md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
