"""Paired evaluation of flip-view averaging on the final output of arbn_teacher (no feedback).

  .venv/bin/python analysis/flip_paired.py   -> analysis/flip_paired.log, analysis/flip_paired_records.csv

The flip forward only changes the prediction (eval mode, no grad, no statistics update, no RNG use), so the
model trajectory is identical with and without it. One run therefore yields both outputs exactly paired:
  base : T3A + LAME on the student's logits / features            (its own T3A instance)
  flip : T3A + LAME on the average of original and flipped views  (its own T3A instance)
The adaptation steps below are copied from AdaptiveRBNTeacher.__call__ (fb_w = 0).
Seeds 0, 1, 2 x official val + the 6 domain orders (short streams, every image once per run).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as E  # noqa: E402
from tta.arbn_teacher import AdaptiveRBNTeacher  # noqa: E402
from tta.arbn_tent import lame  # noqa: E402
from tta.common import ema_update, entropy, symmetric_cross_entropy  # noqa: E402
from tta.t3a import T3A  # noqa: E402


class Paired(AdaptiveRBNTeacher):
    def __init__(self, model, **kw):
        super().__init__(model, **kw)
        self.t3a_flip = T3A(self.model.fc, self.t3a_k, self.NUM_CLASSES)
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
            s_feats = self._feats.detach()
            with torch.no_grad():
                self.model.eval()
                f_out = self.model(torch.flip(x, dims=[3]))
                f_feats = self._feats.detach()
                self.model.train()
            self.optimizer.zero_grad()
            (self.w_ent * entropy(s_out).mean()).backward()
            self.model.eval()
            a_out = self.model(self.transform(x))
            self.model.train()
            (self.w_cons * symmetric_cross_entropy(a_out, t_out).mean()).backward()
            self.optimizer.step()
        ema_update(self.teacher, self.model, self.mt)
        with torch.no_grad():
            s_out = s_out.detach()
            base = lame(self.t3a(s_feats, s_out), s_feats, self.lame_k)
            a_logits, a_feats = (s_out + f_out) / 2, (s_feats + f_feats) / 2
            flip = lame(self.t3a_flip(a_feats, a_logits), a_feats, self.lame_k)
            self.rec.append(dict(base=base.cpu(), flip=flip.cpu(), student=s_out.float().cpu(),
                                 flipped_only=f_out.float().cpu(), avg_student=a_logits.float().cpu()))
        return base.argmax(1)


def rank(scores, y):
    return (scores > scores.gather(1, y[:, None])).sum(1).numpy()


rows = []
for seed in (0, 1, 2):
    for sname, idx in E.ORDERS.items():
        pred, m = E.stream(lambda mm: Paired(mm), idx, seed=seed)
        R = {k: torch.cat([r[k] for r in m.rec]) for k in m.rec[0]}
        y = torch.tensor(E.y[idx])
        rows.append(pd.DataFrame(dict(seed=seed, stream=sname, y=y.numpy(), domain=E.dom[idx],
                                      **{f"rank_{k}": rank(v, y) for k, v in R.items()})))
        print("done", seed, sname, flush=True)
D = pd.concat(rows, ignore_index=True)
D.to_csv(E.ROOT / "analysis/flip_paired_records.csv", index=False)

print("\n=== 1. paired accuracy, same seed + same order (flip minus base, final T3A + LAME output)")
A = D.groupby(["seed", "stream"]).agg(base=("rank_base", lambda r: (r == 0).mean()),
                                       flip=("rank_flip", lambda r: (r == 0).mean())).reset_index()
A["diff"] = A.flip - A.base
print(A.pivot(index="stream", columns="seed", values="diff").round(4).to_string())
print(f"mean diff {A['diff'].mean():+.4f}, min {A['diff'].min():+.4f}, max {A['diff'].max():+.4f}, "
      f"flip better in {(A['diff'] > 0).sum()}/{len(A)} pairs")
for seed in (0, 1, 2):
    a = A[A.seed == seed]; o = a[a.stream != "official"]
    print(f"seed {seed}: official base {a[a.stream == 'official'].base.item():.4f} -> flip "
          f"{a[a.stream == 'official'].flip.item():.4f} | 6-order mean {o.base.mean():.4f} -> {o.flip.mean():.4f} | "
          f"6-order worst {o.base.min():.4f} -> {o.flip.min():.4f}")
print("by domain (all runs):", {d: f"{(g.rank_base == 0).mean():.4f} -> {(g.rank_flip == 0).mean():.4f}"
                                 for d, g in D.groupby("domain")})
print(f"top-3: base {(D.rank_base <= 2).mean():.4f} -> flip {(D.rank_flip <= 2).mean():.4f}")

print("\n=== 2. base G3 errors (truth not in the base final top-3): what does the flip path do with them?")
for name, g in [("all domains", D[D.rank_base > 2]), ("fisheye", D[(D.rank_base > 2) & (D.domain == "fisheye")])]:
    n = len(g)
    print(f"{name}: {n} images ({n / 21:.0f} per run) | flip final: truth enters top-3 {(g.rank_flip <= 2).mean():.1%}, "
          f"reaches top-1 {(g.rank_flip == 0).mean():.1%} | flipped view alone has truth in top-3 "
          f"{(g.rank_flipped_only <= 2).mean():.1%}, top-1 {(g.rank_flipped_only == 0).mean():.1%}")
w = D[D.rank_base > 0]
print(f"\nall base errors {len(w) / 21:.0f}/run: flip fixes {((w.rank_flip == 0).sum()) / 21:.0f}/run; "
      f"flip breaks {((D.rank_base == 0) & (D.rank_flip > 0)).sum() / 21:.0f}/run")
