"""Analyse history_records.csv / history_batches.csv (see history_exp.py)."""
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
D = pd.read_csv("analysis/history_records.csv")
Bt = pd.read_csv("analysis/history_batches.csv")
A = D[D.variant == "A"]
DOMS = ["scanline", "wave", "fisheye"]
bb = lambda b: np.where(b >= 3, "3+", b.astype(str))

print("=== 1. same image, same batch companions, different history (A, aligned orders, 3 seeds)")
first = A[A.seg == 0].groupby(["seed", "img"]).ok_final.mean()          # mean over the 2 orders where it is first
for (dom, prev), g in A[A.seg > 0].groupby(["domain", "prev"]):
    after = g.groupby(["seed", "img", "bidx"]).ok_final.mean().reset_index()   # 1 order per (prev) -> 0/1
    f = first.reindex(pd.MultiIndex.from_frame(after[["seed", "img"]])).values
    after["first"] = f
    after["b"] = bb(after.bidx.values)
    print(f"{dom:9s} after {prev:9s}: first-position acc {np.mean(f):.4f} vs after-shift acc {after.ok_final.mean():.4f} "
          f"| harmed (first right, after wrong) {((f == 1) & (after.ok_final == 0)).mean():.1%}  "
          f"helped (first wrong, after right) {((f == 0) & (after.ok_final == 1)).mean():.1%}  "
          f"always wrong {((f == 0) & (after.ok_final == 0)).mean():.1%}")
    t = after.groupby("b").apply(lambda h: pd.Series(dict(first=h["first"].mean(), after=h.ok_final.mean())),
                                 include_groups=False)
    print("      by batch index after the shift:", {k: f"{v['first']:.3f}->{v['after']:.3f}" for k, v in t.iterrows()})

print("\n=== 2. accuracy by batch index inside a segment (first segment vs later segments)")
for v in ("A", "B"):
    g = D[D.variant == v].assign(b=lambda h: bb(h.bidx.values), pos=lambda h: np.where(h.seg == 0, "first segment", "after a shift"))
    print(v, g.groupby(["pos", "b"]).ok_final.mean().unstack().round(4).to_string())

print("\n=== 3. B (momentum reset) minus A, paired per seed + order + image, by batch index after a shift")
P = D.pivot_table(index=["seed", "order", "img", "seg", "bidx", "domain"], columns="variant", values="ok_final").reset_index()
P["d"] = P.B - P.A
P["b"] = bb(P.bidx.values)
print("after a shift:", P[P.seg > 0].groupby("b").d.mean().round(4).to_dict(),
      "| first segment:", round(P[P.seg == 0].d.mean(), 4), "| overall:", round(P.d.mean(), 4))
print("by domain (after a shift):", P[P.seg > 0].groupby("domain").d.mean().round(4).to_dict())

print("\n=== 4. momentum vs fresh gradient (A): cosine right before optimizer.step()")
a = Bt[(Bt.variant == "A") & Bt.mom_cos.notna()].copy()
a["b"] = bb(a.bidx.values)
print(a[a.seg > 0].groupby("b").mom_cos.describe()[["mean", "25%", "50%", "75%"]].round(3).to_string())
print("first segment, batch>=3:", a[(a.seg == 0) & (a.bidx >= 3)].mom_cos.describe()[["mean", "50%"]].round(3).to_dict())
print("detected onsets (alpha>=0.3 after <0.3) per aligned stream:",
      Bt[Bt.variant == "A"].groupby(["seed", "order"]).apply(
          lambda g: int(((g.alpha >= 0.3) & (g.alpha.shift(1, fill_value=0) < 0.3)).sum()), include_groups=False).value_counts().to_dict())

print("\n=== 5. historical neighbours vs current-batch neighbours (A, batches >= 1, final-label votes)")
h = A[A.hist_vote >= 0]
cw, cr = h[h.cur_vote != h.y], h[h.cur_vote == h.y]
print(f"current-batch vote wrong: {len(cw) / len(h):.1%} of images; history vote right there: {(cw.hist_vote == cw.y).mean():.1%}")
print(f"current-batch vote right: history vote wrong there: {(cr.hist_vote != cr.y).mean():.1%}")
fw = h[~h.ok_final]
print(f"images the final (T3A+LAME) gets wrong: {len(fw) / len(h):.1%}; history vote right on them: {(fw.hist_vote == fw.y).mean():.1%}"
      f" (current-batch vote right on them: {(fw.cur_vote == fw.y).mean():.1%})")
fr = h[h.ok_final]
print(f"images the final gets right: history vote wrong on them: {(fr.hist_vote != fr.y).mean():.1%}")
print(f"  -> per val-sized run (2048 images): history could fix at most {(fw.hist_vote == fw.y).sum() / 18:.0f}, "
      f"and disagrees with a correct final on {(fr.hist_vote != fr.y).sum() / 18:.0f}")

print("\n=== 6. steady part (batch index >= 2), stage by stage: original+flip -> T3A -> LAME (A)")
s = A[A.bidx >= 2]
pat = (s.ok_base.map({True: "R", False: "W"}) + s.ok_t3a.map({True: "R", False: "W"}) + s.ok_final.map({True: "R", False: "W"}))
c = pat.value_counts()
print((c / 18).round(1).rename("images per run").to_string(), f"\n(n per run = {len(s) / 18:.0f}; letters = base, T3A, LAME; R right, W wrong)")
print(f"final accuracy {s.ok_final.mean():.4f}; any stage right {(s.ok_base | s.ok_t3a | s.ok_final).mean():.4f}; "
      f"right before LAME but wrong after: {((s.ok_t3a) & (~s.ok_final)).sum() / 18:.1f}/run; "
      f"base right but T3A wrong: {((s.ok_base) & (~s.ok_t3a)).sum() / 18:.1f}/run")
