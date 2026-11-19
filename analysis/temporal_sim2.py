"""Local class prior (log form) on top of LAME: offline simulation on the cached view records.

  Z_i = softmax( log p_i + (W_knn Z)_i + lam * (K_t Z)_i + beta * log(eps + prior_i) ),
  prior_i = normalized (K_p Z)_i   (soft class counts in a temporal window around image i, self included)

  .venv/bin/python analysis/temporal_sim2.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temporal_sim import knn_affinity, load, time_affinity  # noqa: E402


def stage(run, k=5, lam=0.0, tau=4.0, kind="box", beta=0.0, ptau=8.0, pkind="box", eps=0.02, iters=10, temp=1.0,
          carry_prior=False, pself=True):
    preds = []
    tail = None
    for raw, t3, ft in run["batches"]:
        n = len(t3)
        log_p = (t3 / temp).log_softmax(1)
        W = knn_affinity(ft, k) if k else 0
        if lam > 0:
            W = W + lam * time_affinity(n, tau, kind)
        if beta > 0:
            Kp = time_affinity(n, ptau, pkind)
            if pself:
                Kp = Kp + torch.eye(n, device=Kp.device)
            if carry_prior and tail is not None:  # final probabilities of the previous batch's tail extend the window
                m = len(tail)
                i = torch.arange(n, device=Kp.device).float(); j = torch.arange(-m, 0, device=Kp.device).float()
                d = (i[:, None] - j[None, :]).abs()
                Kc = (d <= ptau).float() if pkind == "box" else torch.exp(-d / ptau)
                ctx = Kc @ tail
            else:
                ctx = 0
        Z = log_p.softmax(1)
        for _ in range(iters):
            u = log_p + (W @ Z if not isinstance(W, int) else 0)
            if beta > 0:
                pr = Kp @ Z + ctx
                pr = pr / pr.sum(1, keepdim=True)
                u = u + beta * torch.log(eps + pr)
            Z = u.softmax(1)
        preds.append(Z.argmax(1).cpu())
        tail = Z[-16:]
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
        line = (f"{n:<64} all {s['all_mean']:.4f}  official {s['official']:.4f}  scan+wave {s['scan_wave']:.4f}  "
                f"fisheye {s['fisheye']:.4f}  paired {100 * d.mean():+.2f} (min {100 * d.min():+.2f}, wins {(d > 0).sum()}/21)")
        print(line, flush=True); rows.append((s["all_mean"], line))

    fmt("LAME k=5 (current)", base)
    fmt("k=5 + 0.25*box(4) linear time kernel", score(runs, lam=0.25, tau=4))
    print("--- log prior only (no linear time kernel)")
    for pkind, ptau in (("box", 4), ("box", 8), ("box", 12), ("box", 16), ("exp", 4), ("exp", 8), ("exp", 16)):
        for beta, eps in itertools.product((0.5, 1.0, 2.0), (0.01, 0.05)):
            fmt(f"k=5 + beta {beta} log(eps {eps} + prior {pkind}({ptau}))", score(runs, beta=beta, ptau=ptau, pkind=pkind, eps=eps))
    print("=== top 12"); [print(l) for _, l in sorted(rows, reverse=True)[:12]]
