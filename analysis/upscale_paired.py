"""Whole-image upscaling as an extra prediction view on top of arbn_teacher (default: no_geom aug, orig + flip).

  .venv/bin/python analysis/upscale_paired.py   -> analysis/upscale_paired.log, analysis/upscale_records.csv

Views U192 / U224: the full 160 x 160 image resized (bilinear) to 192 / 224, nothing cut off (the adaptive average
pool gives the same 2048-d feature); UF = its horizontal flip. Computed after the statistics update and before the SGD
step, in eval mode, without gradients: the adaptation trajectory is identical for every combination, so one run scores
all of them (each with its own T3A). Seeds 0, 1, 2 x official val + the 6 domain orders.
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

up = lambda x, s: F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
VIEWS = {"F": lambda x: torch.flip(x, dims=[3])}
for s in (192, 224):
    VIEWS[f"U{s}"] = lambda x, s=s: up(x, s)
    VIEWS[f"UF{s}"] = lambda x, s=s: torch.flip(up(x, s), dims=[3])
COMBOS = {"OF": ["O", "F"], "OF+U192": ["O", "F", "U192"], "OF+U224": ["O", "F", "U224"],
          "OF+U192+UF192": ["O", "F", "U192", "UF192"], "OF+U224+UF224": ["O", "F", "U224", "UF224"]}
SINGLE = ["O", "F", "U192", "U224"]


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
            self.signal_hist.append(float(s)); self.alpha_hist.append(alpha)
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
            r["student_OF"] = ((outs["O"][0] + outs["F"][0]) / 2).float().cpu()
        self.rec.append(r)
        return r["final_OF"].argmax(1)


def rank(scores, y):
    return (scores > scores.gather(1, y[:, None])).sum(1).numpy()


if __name__ == "__main__":
    rows = []
    for seed in (0, 1, 2):
        for sname, idx in E.ORDERS.items():
            _, m = E.stream(lambda mm: Paired(mm), idx, seed=seed)
            R = {k: torch.cat([r[k] for r in m.rec]) for k in m.rec[0]}
            y = torch.tensor(E.y[idx])
            rows.append(pd.DataFrame(dict(seed=seed, stream=sname, y=y.numpy(), domain=E.dom[idx],
                                          **{f"rank_{k}": rank(v, y) for k, v in R.items()})))
        print("seed", seed, "done", flush=True)
    D = pd.concat(rows, ignore_index=True)
    D.to_csv(E.ROOT / "analysis/upscale_records.csv", index=False)
    RUNS = 21
    print("sanity: final_OF official seed 0 =", round((D[(D.seed == 0) & (D.stream == "official")].rank_final_OF == 0).mean(), 4),
          "(default arbn_teacher, seed 0)")

    print("\n=== 1. combinations (mean of 3 seeds; 6-order mean / worst per seed then averaged)")
    res = []
    for c in COMBOS:
        k = f"rank_final_{c}"
        A = D.groupby(["seed", "stream"])[k].apply(lambda r: (r == 0).mean()).unstack()
        o = A.drop(columns="official")
        res.append(dict(combo=c, official=A["official"].mean(), orders_mean=o.mean(1).mean(), orders_worst=o.min(1).mean(),
                        **{d: (D[D.domain == d][k] == 0).mean() for d in E.DOMS},
                        scan_wave=(D[D.domain != "fisheye"][k] == 0).mean()))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(res).round(4).to_string(index=False))

    print("\n=== 2. paired vs OF (same seed + order) and per-domain wrong->right / right->wrong per run")
    base = D.groupby(["seed", "stream"]).rank_final_OF.apply(lambda r: (r == 0).mean())
    for c in [c for c in COMBOS if c != "OF"]:
        diff = D.groupby(["seed", "stream"])[f"rank_final_{c}"].apply(lambda r: (r == 0).mean()) - base
        print(f"{c:15s} mean {diff.mean():+.4f}  min {diff.min():+.4f}  max {diff.max():+.4f}  better in {(diff > 0).sum()}/{len(diff)}")
        for d in E.DOMS:
            g = D[D.domain == d]
            fix = ((g.rank_final_OF > 0) & (g[f"rank_final_{c}"] == 0)).sum() / RUNS
            brk = ((g.rank_final_OF == 0) & (g[f"rank_final_{c}"] > 0)).sum() / RUNS
            print(f"      {d:9s} wrong->right {fix:5.1f}/run  right->wrong {brk:5.1f}/run  net {fix - brk:+5.1f}")

    print("\n=== 3. single views (student level)")
    hard = D[(D.rank_final_OF > 0) & (D.rank_student_OF > 0)]
    o_r, o_w = D[D.rank_view_O == 0], D[D.rank_view_O > 0]
    for v in SINGLE:
        k = f"rank_view_{v}"
        print(f"  {v:5s} acc {(D[k] == 0).mean():.4f} | right->wrong (of O right) {(o_r[k] > 0).mean():5.1%} | "
              f"wrong->right (of O wrong) {(o_w[k] == 0).mean():5.1%} | hard errors (final + student wrong, "
              f"{len(hard) / RUNS:.0f}/run) rescued to top-1 {(hard[k] == 0).mean():5.1%}, top-3 {(hard[k] <= 2).mean():5.1%} | "
              + " ".join(f"{d} {(D[D.domain == d][k] == 0).mean():.3f}" for d in E.DOMS))
