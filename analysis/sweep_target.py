"""Sweep: flip-view target mixing (FLIP_TGT) x consistency weight (W_CONS) for arbn_teacher (flip default).
Short streams (official + 6 orders), seeds 0/1/2, per-domain and post-shift accuracies.
   .venv/bin/python analysis/sweep_target.py   -> analysis/sweep_target.csv  (analyse with analyze_sweep.py)"""
import itertools, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as E  # noqa: E402
from tta.arbn_teacher import AdaptiveRBNTeacher  # noqa: E402

out = E.ROOT / "analysis/sweep_target.csv"
rows = pd.read_csv(out).to_dict("records") if out.exists() else []
done = {(r["flip_tgt"], r["w_cons"], r["seed"], r["stream"]) for r in rows}
for b, w in itertools.product([0.0, 0.25, 0.5, 0.75, 1.0], [1.0, 0.5, 2.0]):
    t = time.time()
    for seed in (0, 1, 2):
        for s, idx in E.ORDERS.items():
            if (b, w, seed, s) in done:
                continue
            pred, _ = E.stream(lambda m: AdaptiveRBNTeacher(m, flip_tgt=b, w_cons=w), idx, seed=seed)
            ok = pred == E.y[idx]; d = E.dom[idx]
            starts = [i for i in range(1, len(d)) if d[i] != d[i - 1]]
            early = np.zeros(len(d), bool)
            for st in starts:
                early[st:st + 128] = True
            rows.append(dict(flip_tgt=b, w_cons=w, seed=seed, stream=s, acc=ok.mean(),
                             **{dd: ok[d == dd].mean() for dd in E.DOMS},
                             after_shift=ok[early].mean(), steady=ok[~early].mean()))
            pd.DataFrame(rows).to_csv(out, index=False)
    print(f"flip_tgt={b} w_cons={w} done ({time.time() - t:.0f}s)", flush=True)
