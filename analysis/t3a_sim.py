"""T3A-style class prototypes on top of arbn / arbn+TENT, simulated offline.   .venv/bin/python analysis/t3a_sim.py

T3A (Iwasawa & Matsuo, NeurIPS 2021) keeps, per class, a support set of test features pseudo-labelled by the linear
classifier (initialised with the classifier weights), keeps the FILTER_K lowest-entropy supports per class, and
classifies z by  z @ normalize(sum of normalized supports of each class).  It is output-only (does not change the
adaptation), so one recorded run per stream (features, logits, arbn shift signal) is enough to simulate every variant.

Variants:
  filter_k   supports kept per class (lowest entropy)                         5 / 20 / 50 / 100 / all
  stale      how old supports are handled: "keep" | "win10" / "win30" (only supports from the last N batches)
             | "reset" (drop all collected supports when arbn detects a large shift: alpha_t >= 0.3)
  combine    "replace" (T3A output only) | "avg" (average of linear-classifier and T3A probabilities)
  lame       LAME (k=5) on top of the result
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as E  # noqa: E402
from tta.arbn_tent import AdaptiveRBNTent, lame  # noqa: E402
from tta.norm import AdaptiveRobustBN  # noqa: E402

CACHE = E.ROOT / "out" / "_cache"
CACHE.mkdir(parents=True, exist_ok=True)
W_FC = E.BASE.fc.weight.detach().float().cuda()
B_FC = E.BASE.fc.bias.detach().float().cuda()


def record(base_name, sname, idx):
    f = CACHE / f"rec_{base_name}_{sname}.pt"
    if f.exists():
        return torch.load(f)
    feats, logits, alphas = [], [], []
    base = AdaptiveRobustBN if base_name == "arbn" else AdaptiveRBNTent

    class R(base):
        def __call__(s, x):
            box = {}
            h = s.model.avgpool.register_forward_hook(lambda m, i, o: box.__setitem__("f", o.flatten(1).detach()))
            p = super().__call__(x)
            h.remove()
            feats.append(box["f"].half().cpu()); logits.append(s.last_logits.float().cpu()); alphas.append(s.alpha_hist[-1])
            return p

    E.stream((lambda m: R(m)) if base is AdaptiveRobustBN else (lambda m: R(m, lame_k=None)), idx)
    rec = dict(feats=feats, logits=logits, alphas=alphas)
    torch.save(rec, f)
    return rec


class T3A:
    def __init__(self, filter_k, stale):
        self.filter_k, self.stale = filter_k, stale
        self.init_s = W_FC.clone()
        p0 = (W_FC @ W_FC.T + B_FC).softmax(1)
        self.init_y = p0.argmax(1)
        self.init_e = -(p0 * p0.clamp_min(1e-12).log()).sum(1)
        self.s = torch.empty(0, W_FC.shape[1], device="cuda")
        self.y = torch.empty(0, dtype=torch.long, device="cuda")
        self.e = torch.empty(0, device="cuda")
        self.age = torch.empty(0, dtype=torch.long, device="cuda")
        self.t = 0

    def __call__(self, z, logits, alpha):
        if self.stale == "reset" and alpha >= 0.3:
            self.s, self.y, self.e, self.age = self.s[:0], self.y[:0], self.e[:0], self.age[:0]
        p = logits.softmax(1)
        self.s = torch.cat([self.s, z]); self.y = torch.cat([self.y, p.argmax(1)])
        self.e = torch.cat([self.e, -(p * p.clamp_min(1e-12).log()).sum(1)])
        self.age = torch.cat([self.age, torch.full((len(z),), self.t, device="cuda")])
        if self.stale.startswith("win"):
            keep = self.age > self.t - int(self.stale[3:])
            self.s, self.y, self.e, self.age = self.s[keep], self.y[keep], self.e[keep], self.age[keep]
        self.t += 1
        S = torch.cat([self.init_s, self.s]); Y = torch.cat([self.init_y, self.y]); Ent = torch.cat([self.init_e, self.e])
        if self.filter_k is not None:
            keep = torch.zeros(len(S), dtype=torch.bool, device="cuda")
            for c in range(20):
                ic = (Y == c).nonzero().flatten()
                if len(ic):
                    keep[ic[Ent[ic].argsort()[: self.filter_k]]] = True
            S, Y = S[keep], Y[keep]
        proto = F.normalize(F.one_hot(Y, 20).float().T @ F.normalize(S, dim=1), dim=1)
        return z @ proto.T


def simulate(rec, filter_k, stale, combine, use_lame):
    t3a = T3A(filter_k, stale) if stale != "none" else None
    preds = []
    for z, lg, a in zip(rec["feats"], rec["logits"], rec["alphas"]):
        z, lg = z.float().cuda(), lg.cuda()
        if t3a is None:
            out = lg
        else:
            tl = t3a(z, lg, a)
            out = tl if combine == "replace" else ((lg.softmax(1) + tl.softmax(1)) / 2).log()
        preds.append((lame(out, z, 5) if use_lame else out.softmax(1)).argmax(1).cpu())
    return torch.cat(preds).numpy()


streams = dict(E.ORDERS); streams["long"] = E.LONG
rows = []
for base_name in ["arbn", "arbn_tent"]:
    recs = {s: record(base_name, s, idx) for s, idx in streams.items()}
    print("recorded", base_name, flush=True)
    configs = [("none", None, "-", l) for l in (False, True)]
    configs += [(st, fk, cb, l) for st, fk, cb, l in itertools.product(["keep", "win10", "win30", "reset"],
                                                                         [5, 20, 50, 100, None],
                                                                         ["replace", "avg"], [False, True])]
    for st, fk, cb, l in configs:
        accs = {s: (simulate(recs[s], fk, st, cb, l) == E.y[idx]).mean() for s, idx in streams.items()}
        orders = [v for k, v in accs.items() if k not in ("official", "long")]
        rows.append(dict(base=base_name, stale=st, filter_k=fk if fk else "all", combine=cb, lame=l,
                         official=accs["official"], orders_mean=np.mean(orders), orders_worst=np.min(orders),
                         long=accs["long"]))
    pd.DataFrame(rows).to_csv(E.ROOT / "analysis/t3a_sim.csv", index=False)

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
for base_name in ["arbn", "arbn_tent"]:
    d = df[df.base == base_name]
    print(f"\n=== {base_name}: reference (no T3A)")
    print(d[d.stale == "none"].round(4).to_string(index=False))
    print("top 10 by 6-order mean:")
    print(d.sort_values("orders_mean", ascending=False).head(10).round(4).to_string(index=False))
    for k in ["stale", "filter_k", "combine", "lame"]:
        print(d[d.stale != "none"].groupby(k)[["official", "orders_mean", "orders_worst", "long"]].mean().round(4))
