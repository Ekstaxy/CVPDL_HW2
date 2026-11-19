"""Image bank for the consistency step of arbn_teacher: A none / B recent / C balanced / D full RoTTA CSTU bank.
3 seeds x short streams (official + 6 orders); per-image records -> analysis/bank_records.csv
   .venv/bin/python analysis/sweep_bank.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as E  # noqa: E402
from tta.arbn_teacher import AdaptiveRBNTeacher  # noqa: E402

out = E.ROOT / "analysis/bank_records.csv"
frames = [pd.read_csv(out)] if out.exists() else []
done = set(map(tuple, frames[0][["bank", "seed", "stream"]].drop_duplicates().values)) if frames else set()
for bank in ["none", "recent", "balanced", "rotta"]:
    for seed in (0, 1, 2):
        for s, idx in E.ORDERS.items():
            if (bank, seed, s) in done:
                continue
            pred, _ = E.stream(lambda m: AdaptiveRBNTeacher(m, bank=None if bank == "none" else bank), idx, seed=seed)
            d = E.dom[idx]
            starts = [i for i in range(1, len(d)) if d[i] != d[i - 1]]
            early = np.zeros(len(d), bool)
            for st in starts:
                early[st:st + 128] = True
            pos = np.arange(len(idx))
            ncls = np.array([len(np.unique(E.y[idx][b * 64:(b + 1) * 64])) for b in range(len(idx) // 64)])
            frames.append(pd.DataFrame(dict(bank=bank, seed=seed, stream=s, pos=pos, ok=(pred == E.y[idx]),
                                            domain=d, early=early, batch_ncls=ncls[pos // 64])))
            pd.concat(frames, ignore_index=True).to_csv(out, index=False)
        print(bank, seed, "done", flush=True)
