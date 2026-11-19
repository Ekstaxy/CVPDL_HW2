"""Stacking: power-transformed kNN graph + temporal kernel + class-balance adjustment (offline simulation)."""
import itertools, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from temporal_sim import load  # noqa: E402
from temporal_sim4 import score  # noqa: E402

runs = load()
base = score(runs)
rows = []
def fmt(n, s):
    d = s["accs"] - base["accs"]
    line = (f"{n:<60} all {s['all_mean']:.4f}  official {s['official']:.4f}  scan+wave {s['scan_wave']:.4f}  fisheye "
            f"{s['fisheye']:.4f}  paired {100 * d.mean():+.2f} (min {100 * d.min():+.2f}, wins {(d > 0).sum()}/21)")
    print(line, flush=True); rows.append((s["all_mean"], line))
fmt("LAME k=5 (current)", base)
for mode, lam, tau, bal in itertools.product(("", "pow"), (0.0, 0.15, 0.25, 0.35), (4,), (0.0, 0.5, 1.0)):
    if mode == "" and lam == 0 and bal == 0: continue
    fmt(f"mode '{mode}' time {lam}*box({tau}) balance {bal}", score(runs, mode=mode, lam=lam, tau_t=tau, bal=bal))
for tau, lam in ((2, 0.4), (6, 0.15), (8, 0.1)):
    fmt(f"mode 'pow' time {lam}*box({tau}) balance 0.5", score(runs, mode="pow", lam=lam, tau_t=tau, bal=0.5))
print("=== top 10"); [print(l) for _, l in sorted(rows, reverse=True)[:10]]
