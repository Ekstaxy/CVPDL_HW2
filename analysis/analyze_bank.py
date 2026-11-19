"""Analyse analysis/bank_records.csv (image-bank variants vs A = no bank)."""
import numpy as np, pandas as pd
DOMS = ["scanline", "wave", "fisheye"]
R = pd.read_csv("analysis/bank_records.csv")
pd.set_option("display.width", 220)
run = R.groupby(["bank", "seed", "stream"]).ok.mean().unstack(["seed", "stream"])
A = run.loc["none"]
rows = []
for b in ["none", "auggroups4", "aug_soft", "aug_no_geom", "aug_no_color", "aug_weak"]:
    g = R[R.bank == b]
    acc = run.loc[b]
    orders = acc[[c for c in acc.index if c[1] != "official"]]
    per_seed = orders.groupby(level=0)
    diff = acc - A
    rows.append(dict(bank=b, official=acc[[c for c in acc.index if c[1] == "official"]].mean(),
                     orders_mean=per_seed.mean().mean(), orders_worst=per_seed.min().mean(),
                     **{d: g[g.domain == d].ok.mean() for d in DOMS},
                     scan_wave=g[g.domain != "fisheye"].ok.mean(),
                     after_shift=g[g.early].ok.mean(), steady=g[~g.early].ok.mean(),
                     d_mean=diff.mean(), d_min=diff.min(), wins=f"{(diff > 0).sum()}/{len(diff)}"))
print("=== per variant (3 seeds x 7 short streams; d_* = paired difference to A per seed+stream)")
print(pd.DataFrame(rows).round(4).to_string(index=False))

print("\n=== accuracy by number of classes in the batch (true labels, analysis only), difference to A")
R["ncls_bin"] = pd.cut(R.batch_ncls, [0, 8, 11, 20], labels=["<=8 classes", "9-11", ">=12"])
t = R.groupby(["bank", "ncls_bin"], observed=True).ok.mean().unstack()
print((t - t.loc["none"]).round(4).to_string())
print("share of images:", R[R.bank == "none"].ncls_bin.value_counts(normalize=True).round(3).to_dict())

print("\n=== leave-one-domain-out (choose the bank on two domains, score on the third)")
for held in DOMS:
    fit = R[R.domain != held].groupby("bank").ok.mean()
    best = fit.idxmax()
    h = R[R.domain == held].groupby("bank").ok.mean()
    print(f"held-out {held:9s}: chosen {best:8s} -> {h[best]:.4f} vs A {h['none']:.4f} ({h[best] - h['none']:+.4f})")
