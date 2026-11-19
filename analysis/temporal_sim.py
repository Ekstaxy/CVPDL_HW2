"""Offline simulation of output-only stages that exploit the temporal class locality of the stream.

Uses the cached per-view records of the default method (out/_cache/auto_views, 3 seeds x 7 streams) with the
arbn_best view set (O + F + U224 + UF224). T3A is replayed, then different final stages are compared with LAME k=5.

  .venv/bin/python analysis/temporal_sim.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "starter"))
import evalkit as E  # noqa: E402
from tta.t3a import T3A  # noqa: E402

CACHE = ROOT / "out" / "_cache" / "auto_views"
VIEWS = ["O", "F", "U224", "UF224"]
SEEDS = (0, 1, 2)


def load():
    """-> list of runs: dict(seed, stream, y, dom, batches=[(raw_logits, t3a_logits, feats)])"""
    runs = []
    for seed in SEEDS:
        for s, idx in E.ORDERS.items():
            rec = torch.load(CACHE / f"{seed}_{s}.pt", weights_only=False)
            t3a = T3A(E.BASE.fc, 50, 20)
            bs = []
            for b in rec:
                lg = torch.stack([b[v][0].float() for v in VIEWS]).mean(0).cuda()
                ft = torch.stack([b[v][1].float() for v in VIEWS]).mean(0).cuda()
                bs.append((lg, t3a(ft, lg), ft))
            runs.append(dict(seed=seed, stream=s, y=E.y[idx], dom=E.dom[idx], batches=bs))
    return runs


def knn_affinity(feats, k):
    f = F.normalize(feats, dim=1)
    sim = f @ f.T
    knn = sim.topk(k + 1, dim=1).indices[:, 1:]
    W = torch.zeros_like(sim).scatter_(1, knn, 1.0)
    return (W + W.T) / 2


def time_affinity(n, tau, kind="exp", device="cuda"):
    i = torch.arange(n, device=device).float()
    d = (i[:, None] - i[None, :]).abs()
    if kind == "exp":
        W = torch.exp(-d / tau)
    elif kind == "box":
        W = (d <= tau).float()
    else:  # gauss
        W = torch.exp(-0.5 * (d / tau) ** 2)
    W.fill_diagonal_(0)
    return W


def lame_w(logits, W, iters=10):
    log_p = logits.log_softmax(1)
    Z = log_p.softmax(1)
    for _ in range(iters):
        Z = (log_p + W @ Z).softmax(1)
    return Z


def final_stage(run, k=5, lam=0.0, tau=8.0, kind="exp", carry=0, iters=10, temp=1.0, use="t3a", norm_time=False):
    """LAME with affinity  W = kNN(k) + lam * time kernel.  carry > 0: the last `carry` images of the previous batch
    (features + final probabilities, fixed) join the graph as context nodes."""
    preds = []
    prev_f = prev_z = None
    for raw, t3, ft in run["batches"]:
        lg = (t3 if use == "t3a" else raw) / temp
        n = len(lg)
        if carry and prev_f is not None:
            cf, cz = prev_f[-carry:], prev_z[-carry:]
            allf = torch.cat([cf, ft]); m = len(cf)
        else:
            allf, m, cz = ft, 0, None
        W = knn_affinity(allf, k) if k else torch.zeros(len(allf), len(allf), device=ft.device)
        if lam > 0:
            Wt = time_affinity(len(allf), tau, kind)
            if norm_time:
                Wt = Wt / Wt.sum(1, keepdim=True).clamp_min(1e-8)
            W = W + lam * Wt
        log_p = lg.log_softmax(1)
        Z = log_p.softmax(1)
        for _ in range(iters):
            full = torch.cat([cz, Z]) if m else Z
            Z = (log_p + (W @ full)[m:]).softmax(1)
        preds.append(Z.argmax(1).cpu())
        prev_f, prev_z = ft, Z
    return torch.cat(preds).numpy()


def score(runs, **kw):
    acc, off, sw_c, sw_n, fi_c, fi_n = [], [], 0, 0, 0, 0
    for r in runs:
        ok = final_stage(r, **kw) == r["y"]
        acc.append(ok.mean())
        if r["stream"] == "official":
            off.append(ok.mean())
        m = r["dom"] != "fisheye"
        sw_c += ok[m].sum(); sw_n += m.sum(); fi_c += ok[~m].sum(); fi_n += (~m).sum()
    return dict(all_mean=np.mean(acc), official=np.mean(off), scan_wave=sw_c / sw_n, fisheye=fi_c / fi_n,
                accs=np.array(acc))


def oracle(runs):
    """Upper bounds: mask the T3A logits to the classes truly present in the batch / in a +-w window."""
    out = {}
    for name, w in (("batch classes", None), ("window +-8", 8), ("window +-4", 4)):
        acc = []
        for r in runs:
            y = r["y"]; p = []
            for b, (raw, t3, ft) in enumerate(r["batches"]):
                yb = y[b * 64:b * 64 + len(t3)]
                mask = torch.full_like(t3, -1e9)
                for i in range(len(t3)):
                    cls = np.unique(yb if w is None else yb[max(0, i - w):i + w + 1])
                    mask[i, torch.as_tensor(cls, device=t3.device)] = 0
                p.append((t3 + mask).argmax(1).cpu())
            acc.append((torch.cat(p).numpy() == y).mean())
        out[name] = np.mean(acc)
    return out


if __name__ == "__main__":
    runs = load()
    base = score(runs)
    fmt = lambda n, s: print(f"{n:<52} all {s['all_mean']:.4f}  official {s['official']:.4f}  scan+wave {s['scan_wave']:.4f}  "
                             f"fisheye {s['fisheye']:.4f}  paired {100 * (s['accs'] - base['accs']).mean():+.2f} "
                             f"(min {100 * (s['accs'] - base['accs']).min():+.2f}, wins {(s['accs'] > base['accs']).sum()}/21)")
    fmt("LAME k=5 (current)", base)
    fmt("no LAME (T3A argmax)", score(runs, k=0, iters=0))
    print("oracle masks on T3A logits:", {k: round(v, 4) for k, v in oracle(runs).items()})
    print("--- time kernel added to kNN (k=5)")
    for kind, tau in itertools.product(("exp", "box"), (2, 4, 8, 16)):
        for lam in (0.1, 0.25, 0.5, 1.0):
            fmt(f"k=5 + {lam} * {kind}(tau={tau})", score(runs, lam=lam, tau=tau, kind=kind))
    print("--- carry-over context from the previous batch")
    for carry in (8, 16, 32, 64):
        fmt(f"k=5, carry {carry}", score(runs, carry=carry))
