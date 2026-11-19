"""Center-crop as an extra prediction view on top of arbn_teacher (original + flip), exactly paired.

  .venv/bin/python analysis/crop_paired.py   -> analysis/crop_paired.log, analysis/crop_paired_records.csv

All extra views (flip, center crops resized back to 160, flipped crops) are computed in eval mode after the
statistics update and before the SGD step: they never touch the BN statistics, the parameters or the RNG, so the
model trajectory is the same for every view combination and one run scores all of them, each with its own T3A.
Seeds 0, 1, 2 x official val + the 6 domain orders (short streams, every image once per run).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as E  # noqa: E402
from tta.arbn_teacher import AdaptiveRBNTeacher  # noqa: E402
from tta.arbn_tent import lame  # noqa: E402
from tta.common import ema_update, entropy, symmetric_cross_entropy  # noqa: E402
from tta.t3a import T3A  # noqa: E402


def crop(x, r):
    s = int(round(x.shape[-1] * r)); o = (x.shape[-1] - s) // 2
    return F.interpolate(x[:, :, o:o + s, o:o + s], size=x.shape[-2:], mode="bilinear", align_corners=False)


VIEWS = {"F": lambda x: torch.flip(x, dims=[3])}
for r in (0.9, 0.8, 0.7):
    VIEWS[f"C{int(r * 100)}"] = lambda x, r=r: crop(x, r)
    VIEWS[f"CF{int(r * 100)}"] = lambda x, r=r: torch.flip(crop(x, r), dims=[3])
COMBOS = {"O": ["O"], "OF": ["O", "F"],
          "OF+C90": ["O", "F", "C90"], "OF+C80": ["O", "F", "C80"], "OF+C70": ["O", "F", "C70"],
          "OF+C80+CF80": ["O", "F", "C80", "CF80"], "OF+C70+CF70": ["O", "F", "C70", "CF70"]}
SINGLE = ["O", "F", "C90", "C80", "C70"]


class Paired(AdaptiveRBNTeacher):
    def __init__(self, model, **kw):
        super().__init__(model, flip=False, **kw)
        self.t3as = {c: T3A(self.model.fc, self.t3a_k, self.NUM_CLASSES) for c in COMBOS}
        self.rec = []

    def __call__(self, x):
        with torch.no_grad():
            s = self._shallow_signal(x)
            alpha = float((self.gain * s).clamp(self.alpha_min, self.alpha_max))
            for l in self.layers + self.t_layers:
                l.momentum = alpha
            self.teacher.train()
            t_out = self.teacher(x)
        self.model.train()
        with torch.enable_grad():
            s_out = self.model(x)
            outs = {"O": (s_out.detach(), self._feats.detach())}
            with torch.no_grad():
                self.model.eval()
                for v, fn in VIEWS.items():
                    o = self.model(fn(x)); outs[v] = (o, self._feats.detach())
                self.model.train()
            self.optimizer.zero_grad()
            (self.w_ent * entropy(s_out).mean()).backward()
            self.model.eval()
            a_out = self.model(self.transform(x))
            self.model.train()
            (self.w_cons * symmetric_cross_entropy(a_out, t_out).mean()).backward()
            self.optimizer.step()
        ema_update(self.teacher, self.model, self.mt)
        r = {}
        with torch.no_grad():
            for c, vs in COMBOS.items():
                lg = torch.stack([outs[v][0] for v in vs]).mean(0)
                ft = torch.stack([outs[v][1] for v in vs]).mean(0)
                r[f"final_{c}"] = lame(self.t3as[c](ft, lg), ft, self.lame_k).cpu()
            for v in SINGLE:
                r[f"view_{v}"] = outs[v][0].float().cpu()
        self.rec.append(r)
        return r["final_OF"].argmax(1)


def rank(scores, y):
    return (scores > scores.gather(1, y[:, None])).sum(1).numpy()


rows = []
for seed in (0, 1, 2):
    for sname, idx in E.ORDERS.items():
        _, m = E.stream(lambda mm: Paired(mm), idx, seed=seed)
        R = {k: torch.cat([r[k] for r in m.rec]) for k in m.rec[0]}
        y = torch.tensor(E.y[idx])
        rows.append(pd.DataFrame(dict(seed=seed, stream=sname, y=y.numpy(), domain=E.dom[idx],
                                      **{f"rank_{k}": rank(v, y) for k, v in R.items()})))
        print("done", seed, sname, flush=True)
D = pd.concat(rows, ignore_index=True)
D.to_csv(E.ROOT / "analysis/crop_paired_records.csv", index=False)
RUNS = 21

print("=== sanity: final_OF official seed 0 should equal the arbn_teacher(flip) run:",
      round((D[(D.seed == 0) & (D.stream == 'official')].rank_final_OF == 0).mean(), 4))

print("\n=== 1. accuracy of every combination (mean over 3 seeds; 6-order mean / worst per seed then averaged)")
res = []
for c in COMBOS:
    k = f"rank_final_{c}"
    A = D.groupby(["seed", "stream"])[k].apply(lambda r: (r == 0).mean()).unstack()
    off = A["official"]; o = A.drop(columns="official")
    res.append(dict(combo=c, official=off.mean(), orders_mean=o.mean(1).mean(), orders_worst=o.min(1).mean(),
                    **{d: (D[D.domain == d][k] == 0).mean() for d in E.DOMS}, top3=(D[k] <= 2).mean()))
print(pd.DataFrame(res).round(4).to_string(index=False))

print("\n=== 2. paired vs current (OF), same seed + same order: accuracy difference")
base = D.groupby(["seed", "stream"]).rank_final_OF.apply(lambda r: (r == 0).mean())
for c in [c for c in COMBOS if c not in ("O", "OF")]:
    diff = D.groupby(["seed", "stream"])[f"rank_final_{c}"].apply(lambda r: (r == 0).mean()) - base
    print(f"{c:12s} mean {diff.mean():+.4f}  min {diff.min():+.4f}  max {diff.max():+.4f}  better in {(diff > 0).sum()}/{len(diff)}")
    for d in E.DOMS:
        g = D[D.domain == d]
        fix = ((g.rank_final_OF > 0) & (g[f"rank_final_{c}"] == 0)).sum() / RUNS
        brk = ((g.rank_final_OF == 0) & (g[f"rank_final_{c}"] > 0)).sum() / RUNS
        print(f"      {d:9s} wrong->right {fix:5.1f}/run  right->wrong {brk:5.1f}/run  net {fix - brk:+5.1f}")

print("\n=== 3. single views on the hard errors (student level, truth in top-3 / top-1)")
for pool_name, pool in [("base G3 (O-final, truth outside top-3)", D.rank_final_O > 2),
                        ("current G3 (OF-final, truth outside top-3)", D.rank_final_OF > 2)]:
    for dname, dm in [("all", np.ones(len(D), bool)), ("fisheye", D.domain == "fisheye")]:
        g = D[pool & dm]
        print(f"{pool_name}, {dname}: {len(g) / RUNS:.0f}/run  " + "  ".join(
            f"{v}: {(g[f'rank_view_{v}'] <= 2).mean():.1%}/{(g[f'rank_view_{v}'] == 0).mean():.1%}" for v in SINGLE))
print("single-view accuracy by domain:")
for v in SINGLE:
    print(f"  {v:4s} " + "  ".join(f"{d} {(D[D.domain == d][f'rank_view_{v}'] == 0).mean():.4f}" for d in E.DOMS))
