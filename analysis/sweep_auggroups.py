"""B = strong augmentation drawn per group of 16 (4 groups) instead of once per batch; standard short streams,
3 seeds x 7, appended to bank_records.csv as bank='auggroups4' (compare with A = 'none' via analyze_bank.py)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as E  # noqa: E402
from tta.arbn_teacher import AdaptiveRBNTeacher  # noqa: E402
out = E.ROOT / "analysis/bank_records.csv"
R = pd.read_csv(out); frames = [R[R.bank != "auggroups4"]]
for seed in (0, 1, 2):
    for s, idx in E.ORDERS.items():
        pred, m = E.stream(lambda mm: AdaptiveRBNTeacher(mm, aug_groups=4), idx, seed=seed)
        d = E.dom[idx]
        starts = [i for i in range(1, len(d)) if d[i] != d[i - 1]]
        early = np.zeros(len(d), bool)
        for st in starts:
            early[st:st + 128] = True
        pos = np.arange(len(idx))
        ncls = np.array([len(np.unique(E.y[idx][b * 64:(b + 1) * 64])) for b in range(len(idx) // 64)])
        frames.append(pd.DataFrame(dict(bank="auggroups4", seed=seed, stream=s, pos=pos, ok=(pred == E.y[idx]),
                                        domain=d, early=early, batch_ncls=ncls[pos // 64])))
    print("seed", seed, "done", flush=True)
pd.concat(frames, ignore_index=True).to_csv(out, index=False)
print("saved")
