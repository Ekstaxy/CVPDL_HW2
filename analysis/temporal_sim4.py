"""Graph quality for LAME: feature transforms for the kNN affinity (centering, power transform), kNN purity,
unary temperature / graph weight, stacked with the temporal kernel and the class-balance adjustment.

  .venv/bin/python analysis/temporal_sim4.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temporal_sim import load, time_affinity  # noqa: E402


def transform(ft, mode, state):
    """Feature transform before cosine kNN. state: dict holding a running mean (EMA over batches)."""
    f = ft
    if "pow" in mode:
        f = f.clamp_min(0).pow(0.5)
    if "bcenter" in mode:        # center with the batch mean
        f = f - f.mean(0, keepdim=True)
    elif "rcenter" in mode:      # center with a running mean (EMA 0.8 over batches, includes this batch)
        mu = f.mean(0, keepdim=True)
        state["mu"] = mu if state.get(mode) is None else 0.8 * state[mode] + 0.2 * mu
        state[mode] = state["mu"]
        f = f - state["mu"]
    return F.normalize(f, dim=1)


def affinity(f, k, soft=None, mutual=False, tbonus=0.0):
    sim = f @ f.T
    if tbonus > 0:               # temporal bonus inside the neighbour search
        i = torch.arange(len(f), device=f.device).float()
        sim = sim + tbonus * torch.exp(-(i[:, None] - i[None, :]).abs() / 4.0)
    sim.fill_diagonal_(-1e9)
    val, knn = sim.topk(k, dim=1)
    W = torch.zeros_like(sim).scatter_(1, knn, 1.0 if soft is None else torch.exp((val - val[:, :1]) / soft))
    return torch.minimum(W, W.T) * 2 if mutual else (W + W.T) / 2


def stage(run, k=5, mode="", lam=0.0, tau_t=4.0, bal=0.0, rho=0.95, n0=320.0, temp=1.0, wscale=1.0, iters=10,
          soft=None, mutual=False, tbonus=0.0, purity=None):
    preds, K, state = [], 20, {}
    cnt = torch.full((K,), n0 / K, device="cuda")
    off = 0
    for raw, t3, ft in run["batches"]:
        n = len(t3)
        bias = -bal * torch.log(K * cnt / cnt.sum()) if bal > 0 else 0
        log_p = (t3 / temp + bias).log_softmax(1)
        f = transform(ft, mode, state)
        W = affinity(f, k, soft, mutual, tbonus)
        if purity is not None:
            yb = torch.as_tensor(run["y"][off:off + n], device=ft.device)
            nb = (W > 0).float()
            purity.append(float(((yb[:, None] == yb[None, :]).float() * nb).sum() / nb.sum()))
        W = wscale * W
        if lam > 0:
            W = W + lam * time_affinity(n, tau_t, "box")
        Z = log_p.softmax(1)
        for _ in range(iters):
            Z = (log_p + W @ Z).softmax(1)
        preds.append(Z.argmax(1).cpu())
        cnt = rho * cnt + Z.sum(0)
        off += n
    return torch.cat(preds).numpy()


def score(runs, **kw):
    acc, off, sw_c, sw_n, fi_c, fi_n, pur = [], [], 0, 0, 0, 0, []
    for r in runs:
        ok = stage(r, purity=pur, **kw) == r["y"]
        acc.append(ok.mean())
        if r["stream"] == "official":
            off.append(ok.mean())
        m = r["dom"] != "fisheye"
        sw_c += ok[m].sum(); sw_n += m.sum(); fi_c += ok[~m].sum(); fi_n += (~m).sum()
    return dict(all_mean=np.mean(acc), official=np.mean(off), scan_wave=sw_c / sw_n, fisheye=fi_c / fi_n,
                accs=np.array(acc), purity=np.mean(pur))


if __name__ == "__main__":
    runs = load()
    base = score(runs)
    rows = []

    def fmt(n, s):
        d = s["accs"] - base["accs"]
        line = (f"{n:<60} all {s['all_mean']:.4f}  official {s['official']:.4f}  scan+wave {s['scan_wave']:.4f}  fisheye "
                f"{s['fisheye']:.4f}  purity {s['purity']:.3f}  paired {100 * d.mean():+.2f} (min {100 * d.min():+.2f}, wins {(d > 0).sum()}/21)")
        print(line, flush=True); rows.append((s["all_mean"], line))

    fmt("LAME k=5 (current)", base)
    print("--- feature transform for the kNN graph")
    for mode in ("bcenter", "rcenter", "pow", "pow+bcenter", "pow+rcenter"):
        for k in (5, 7):
            fmt(f"{mode}, k={k}", score(runs, mode=mode, k=k))
    print("--- graph construction")
    fmt("mutual kNN k=8", score(runs, k=8, mutual=True))
    fmt("soft weights 0.1, k=8", score(runs, k=8, soft=0.1))
    for tb in (0.02, 0.05, 0.1):
        fmt(f"temporal bonus {tb} in neighbour search, k=5", score(runs, tbonus=tb))
    print("--- unary temperature / graph weight")
    for temp, ws in itertools.product((0.5, 1.0, 2.0), (0.5, 1.0, 2.0)):
        if (temp, ws) != (1.0, 1.0):
            fmt(f"temp {temp}, graph weight {ws}", score(runs, temp=temp, wscale=ws))
    print("=== top 10"); [print(l) for _, l in sorted(rows, reverse=True)[:10]]
