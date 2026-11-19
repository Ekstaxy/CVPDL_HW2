"""Analyse analysis/sweep_target.csv: per-config table, paired differences vs the baseline, leave-one-domain-out."""
import sys
import numpy as np, pandas as pd
DOMS = ["scanline", "wave", "fisheye"]
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "analysis/sweep_target.csv")
df["cfg"] = list(zip(df.flip_tgt, df.w_cons))
BASE = (0.0, 1.0)
complete = [c for c, g in df.groupby("cfg") if len(g) == 21]
df = df[df.cfg.isin(complete)]
base = df[df.cfg == BASE].set_index(["seed", "stream"])
rows = []
for c, g in df.groupby("cfg"):
    gi = g.set_index(["seed", "stream"])
    o = g[g.stream != "official"].groupby("seed").acc
    diff = gi.acc - base.acc
    rows.append(dict(flip_tgt=c[0], w_cons=c[1], official=g[g.stream == "official"].acc.mean(),
                     orders_mean=o.mean().mean(), orders_worst=o.min().mean(),
                     **{d: g[d].mean() for d in DOMS}, scan_wave=(g.scanline.mean() + g.wave.mean()) / 2,
                     after_shift=g.after_shift.mean(), steady=g.steady.mean(),
                     d_mean=diff.mean(), d_min=diff.min(), wins=f"{(diff > 0).sum()}/{len(diff)}",
                     d_scan_wave=((gi.scanline - base.scanline) + (gi.wave - base.wave)).mean() / 2))
T = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("=== per config (mean of 3 seeds; d_* = paired difference to flip_tgt=0, w_cons=1)")
print(T.sort_values(["flip_tgt", "w_cons"]).round(4).to_string(index=False))
print("\n=== leave-one-domain-out: choose the config on two domains, score it on the held-out domain")
for held in DOMS:
    fit = [d for d in DOMS if d != held]
    best = max(complete, key=lambda c: df[df.cfg == c][fit].mean().mean())
    b, gb = df[df.cfg == best], df[df.cfg == BASE]
    print(f"held-out {held:9s}: chosen flip_tgt={best[0]} w_cons={best[1]}  held-out acc {b[held].mean():.4f} "
          f"vs baseline {gb[held].mean():.4f}  ({b[held].mean() - gb[held].mean():+.4f})")
