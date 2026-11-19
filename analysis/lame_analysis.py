"""When does LAME help?   .venv/bin/python analysis/lame_analysis.py

LAME only changes the outputs and never feeds back into the adaptation, so one recorded run per stream
(logits + avgpool features per batch) gives both "without LAME" and "with LAME" exactly.

1. 2x2 table: {arbn, arbn+TENT} x {no LAME, LAME}  on official val, 6 domain orders, long stream.
2. For every prediction LAME changes: wrong->right vs right->wrong, broken down by
     margin    = top-1 minus top-2 probability of the base model (how hesitant the base model is)
     agreement = share of the k nearest neighbours (same kNN graph as LAME) whose base prediction is
                 the neighbours' majority class (how consistent the neighbours are)
3. Gated LAME: take LAME's answer only if margin < tm and agreement >= ta. Thresholds are picked by
   leave-one-domain-out (fit on two val domains, score on the third) to limit overfitting.
"""
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

K = 5
OUT = E.ROOT / "analysis"


def record(base, idx):
    rec = []

    class R(base):
        def __call__(s, x):
            feats = {}
            h = s.model.avgpool.register_forward_hook(lambda m, i, o: feats.__setitem__("f", o.flatten(1).detach()))
            p = super().__call__(x)
            h.remove()
            logits = s.last_logits.float()
            q = lame(logits, feats["f"], K).cpu()
            f = F.normalize(feats["f"].float(), dim=1)
            knn = (f @ f.T).topk(K + 1, dim=1).indices[:, 1:].cpu()
            rec.append((logits.softmax(1).cpu(), q, knn))
            return p

    make = (lambda m: R(m)) if base is AdaptiveRobustBN else (lambda m: R(m, lame_k=None))
    E.stream(make, idx)
    rows = []
    for b, (p, q, knn) in enumerate(rec):
        base_pred = p.argmax(1)
        top2 = p.topk(2, dim=1).values
        nb_pred = base_pred[knn]                      # (B, K) neighbours' base predictions
        maj = torch.mode(nb_pred, dim=1).values
        agree = (nb_pred == maj[:, None]).float().mean(1)
        for i in range(len(p)):
            rows.append((b, int(base_pred[i]), int(q[i].argmax()), float(top2[i, 0] - top2[i, 1]),
                         float(agree[i]), int(maj[i])))
    df = pd.DataFrame(rows, columns=["batch", "base", "lame", "margin", "agree", "nb_major"])
    df["y"], df["domain"] = E.y[idx], E.dom[idx]
    return df


def acc(df, col):
    return (df[col] == df.y).mean()


streams = {k: v for k, v in E.ORDERS.items()}
streams["long"] = E.LONG
bases = {"arbn": AdaptiveRobustBN, "arbn+TENT": AdaptiveRBNTent}
data = {}
for bname, base in bases.items():
    for sname, idx in streams.items():
        data[(bname, sname)] = record(base, idx)
        print("recorded", bname, sname, flush=True)

# 1. 2x2 table
print("\n=== 2x2 (official / 6-order mean / 6-order worst / long)")
table = []
for bname in bases:
    for col, lab in [("base", "no LAME"), ("lame", "LAME")]:
        orders = [acc(data[(bname, s)], col) for s in streams if s not in ("official", "long")]
        table.append(dict(base=bname, lame=lab, official=acc(data[(bname, "official")], col),
                          orders_mean=np.mean(orders), orders_worst=np.min(orders),
                          long=acc(data[(bname, "long")], col),
                          **{d: acc(data[(bname, "official")][data[(bname, "official")].domain == d], col) for d in E.DOMS}))
print(pd.DataFrame(table).round(4).to_string(index=False))

# 2. what does LAME change? pooled official + 6 orders (short streams), per base
for bname in bases:
    pool = pd.concat([data[(bname, s)] for s in streams if s != "long"], ignore_index=True)
    ch = pool[pool.base != pool.lame].copy()
    ch["fix"] = (ch.lame == ch.y) & (ch.base != ch.y)
    ch["break"] = (ch.base == ch.y) & (ch.lame != ch.y)
    print(f"\n=== {bname}: LAME changes {len(ch)} / {len(pool)} predictions ({len(ch) / len(pool):.1%}); "
          f"wrong->right {ch.fix.sum()}, right->wrong {ch['break'].sum()}, wrong->wrong {(~ch.fix & ~ch['break']).sum()}")
    ch["margin_bin"] = pd.cut(ch.margin, [0, 0.05, 0.1, 0.2, 0.4, 1.0], include_lowest=True)
    ch["agree_bin"] = pd.cut(ch.agree, [0, 0.41, 0.61, 0.81, 1.0], include_lowest=True,
                             labels=["<=2/5", "3/5", "4/5", "5/5"])
    g = ch.groupby(["margin_bin", "agree_bin"], observed=True).agg(n=("fix", "size"), fix=("fix", "sum"),
                                                                  brk=("break", "sum"))
    g["net"] = g.fix - g.brk
    print(g.to_string())
    print("by margin only:")
    print(ch.groupby("margin_bin", observed=True).agg(n=("fix", "size"), fix=("fix", "sum"), brk=("break", "sum")).to_string())
    print("by neighbour agreement only:")
    print(ch.groupby("agree_bin", observed=True).agg(n=("fix", "size"), fix=("fix", "sum"), brk=("break", "sum")).to_string())
    pool.to_csv(OUT / f"lame_changes_{bname.replace('+', '_')}.csv", index=False)

# 3. gated LAME with leave-one-domain-out threshold selection
TM = [0.05, 0.1, 0.2, 0.3, 0.5, 2.0]
TA = [0.0, 0.4, 0.6, 0.8, 1.0]


def gated(df, tm, ta):
    use = (df.margin < tm) & (df.agree >= ta)
    return np.where(use, df.lame, df.base)


print("\n=== gated LAME, leave-one-domain-out (pooled official + 6 orders)")
for bname in bases:
    pool = pd.concat([data[(bname, s)] for s in streams if s != "long"], ignore_index=True)
    res = []
    for d in E.DOMS:
        fit, test = pool[pool.domain != d], pool[pool.domain == d]
        best = max(((tm, ta) for tm in TM for ta in TA), key=lambda t: (gated(fit, *t) == fit.y).mean())
        res.append(dict(held_out=d, tm=best[0], ta=best[1], base=acc(test, "base"), lame=acc(test, "lame"),
                        gated=(gated(test, *best) == test.y).mean()))
    r = pd.DataFrame(res)
    print(bname); print(r.round(4).to_string(index=False))
    print(f"  mean over held-out domains: base {r.base.mean():.4f}  LAME {r.lame.mean():.4f}  gated {r.gated.mean():.4f}")
    long = data[(bname, "long")]
    for tm, ta in sorted({(x.tm, x.ta) for x in r.itertuples()}):
        print(f"  long stream with tm={tm}, ta={ta}: {(gated(long, tm, ta) == long.y).mean():.4f} "
              f"(base {acc(long, 'base'):.4f}, LAME {acc(long, 'lame'):.4f})")
