"""Long-run class balance (the stream is class-balanced per domain) as an online logit adjustment, with and without
the temporal kernel. Offline simulation on the cached view records.

  bias_c = -tau * log(K * m_c),  m = running class marginal of the final answers (pseudo-count n0 of uniform,
  exponential forgetting rho per batch); added to the T3A logits before LAME.

  .venv/bin/python analysis/temporal_sim3.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temporal_sim import knn_affinity, load, time_affinity  # noqa: E402


def stage(run, k=5, lam=0.0, tau_t=4.0, bal=0.0, rho=1.0, n0=64.0, hard=False, iters=10, where="pre", ret_probs=False):
    preds, K = [], 20
    cnt = torch.full((K,), n0 / K, device="cuda")
    for raw, t3, ft in run["batches"]:
        n = len(t3)
        m = cnt / cnt.sum()
        bias = -bal * torch.log(K * m) if bal > 0 else 0
        log_p = (t3 + (bias if where == "pre" else 0)).log_softmax(1)
        W = knn_affinity(ft, k)
        if lam > 0:
            W = W + lam * time_affinity(n, tau_t, "box")
        Z = log_p.softmax(1)
        for _ in range(iters):
            Z = (log_p + W @ Z).softmax(1)
        if where == "post" and bal > 0:
            Z = (Z.clamp_min(1e-8).log() + bias).softmax(1)
        preds.append(Z.argmax(1).cpu())
        add = torch.nn.functional.one_hot(Z.argmax(1), K).float().sum(0) if hard else Z.sum(0)
        cnt = rho * cnt + add
    return torch.cat(preds).numpy()


def score(runs, **kw):
    acc, off, sw_c, sw_n, fi_c, fi_n = [], [], 0, 0, 0, 0
    for r in runs:
        ok = stage(r, **kw) == r["y"]
        acc.append(ok.mean())
        if r["stream"] == "official":
            off.append(ok.mean())
        m = r["dom"] != "fisheye"
        sw_c += ok[m].sum(); sw_n += m.sum(); fi_c += ok[~m].sum(); fi_n += (~m).sum()
    return dict(all_mean=np.mean(acc), official=np.mean(off), scan_wave=sw_c / sw_n, fisheye=fi_c / fi_n,
                accs=np.array(acc))


if __name__ == "__main__":
    runs = load()
    base = score(runs)
    rows = []

    def fmt(n, s):
        d = s["accs"] - base["accs"]
        line = (f"{n:<58} all {s['all_mean']:.4f}  official {s['official']:.4f}  scan+wave {s['scan_wave']:.4f}  "
                f"fisheye {s['fisheye']:.4f}  paired {100 * d.mean():+.2f} (min {100 * d.min():+.2f}, wins {(d > 0).sum()}/21)")
        print(line, flush=True); rows.append((s["all_mean"], line))

    fmt("LAME k=5 (current)", base)
    fmt("time kernel 0.25*box(4)", score(runs, lam=0.25))
    print("--- balance only")
    for bal, rho, n0 in itertools.product((0.25, 0.5, 1.0, 2.0), (1.0, 0.95, 0.9), (64.0, 320.0)):
        fmt(f"balance tau {bal} rho {rho} n0 {n0:.0f}", score(runs, bal=bal, rho=rho, n0=n0))
    print("=== top 8"); [print(l) for _, l in sorted(rows, reverse=True)[:8]]
