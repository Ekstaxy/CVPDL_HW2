"""Label-free health check of test predictions (run before every submission).

  .venv/bin/python analysis/health_check.py out/<version>/test_pred.csv [reference_csv]

Flags collapse on the long test stream: predicted classes per batch (first vs second half), largest class share,
and agreement with a reference submission per batch. A healthy run keeps ~10-12 classes per batch in both halves
and a largest class share near 1/20.
"""
import sys
import numpy as np, pandas as pd

f = sys.argv[1]
ref = sys.argv[2] if len(sys.argv) > 2 else "out/arbn_teacher_flip/test_pred.csv"
p = pd.read_csv(f).label.values
nb = len(p) // 64
u = np.array([len(np.unique(p[i * 64:(i + 1) * 64])) for i in range(nb)])
share = np.bincount(p, minlength=20).max() / len(p)
print(f"{f}: classes/batch 1st half {u[:nb // 2].mean():.1f}, 2nd half {u[nb // 2:].mean():.1f}, min {u.min()} | "
      f"largest class share {share:.3f}")
if ref != f:
    r = pd.read_csv(ref).label.values
    ag = (p == r).reshape(-1, 64).mean(1)
    print(f"  agreement with {ref}: {np.mean(p == r):.3f} (lowest batch {ag.min():.2f} at batch {ag.argmin()})")
ok = u[nb // 2:].mean() > 0.8 * u[:nb // 2].mean() and share < 0.12
print("  HEALTHY" if ok else "  !!! LOOKS COLLAPSED - do not submit")
