"""Error analysis of the final arbn_teacher (T3A + LAME), offline with val labels (labels never reach the model).

  .venv/bin/python analysis/error_analysis.py   -> analysis/error_analysis.log, analysis/error_records.csv

Per image (official val + the 6 domain orders, seed 0) it records: student logits, T3A logits, final LAME
probabilities, LAME applied to the student alone, the kNN neighbours' majority, and the student's logits on the
horizontally flipped image (same model state: computed after the statistics update, before the SGD step, in eval
mode, so the stream and the submitted predictions are unchanged).
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

CLS = [l.split()[2] for l in open(E.ROOT / "starter/classes.txt")]


class Diag(AdaptiveRBNTeacher):
    """Same computation as AdaptiveRBNTeacher.__call__, plus diagnostics that do not touch the RNG or the state."""

    def __init__(self, model, **kw):
        super().__init__(model, **kw)
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
            with torch.no_grad():  # diagnostic: flipped view, same state as s_out
                self.model.eval()
                f_out = self.model(torch.flip(x, dims=[3]))
                self.model.train()
                self._feats = s_feats
            self.optimizer.zero_grad()
            (self.w_ent * entropy(s_out).mean()).backward()
            self.model.eval()
            a_out = self.model(self.transform(x))
            self.model.train()
            (self.w_cons * symmetric_cross_entropy(a_out, t_out).mean()).backward()
            self.optimizer.step()
        ema_update(self.teacher, self.model, self.mt)
        s_out = s_out.detach()
        t3a_logits = self.t3a(s_feats, s_out)
        final = lame(t3a_logits, s_feats, self.lame_k)
        lame_only = lame(s_out, s_feats, self.lame_k)
        f = F.normalize(s_feats.float(), dim=1)
        knn = (f @ f.T).topk(self.lame_k + 1, dim=1).indices[:, 1:]
        nb_major = torch.mode(s_out.argmax(1)[knn], dim=1).values
        self.rec.append(dict(student=s_out.float().cpu(), t3a=t3a_logits.float().cpu(), final=final.cpu(),
                             lame_only=lame_only.cpu(), nb_major=nb_major.cpu(), flip=f_out.float().cpu()))
        self.last_logits = s_out
        return final.argmax(1)


rows = []
for sname, idx in E.ORDERS.items():
    pred, m = E.stream(lambda mm: Diag(mm), idx)
    R = {k: torch.cat([r[k] for r in m.rec]) for k in m.rec[0]}
    y = torch.tensor(E.y[idx])
    final_rank = (R["final"] > R["final"].gather(1, y[:, None])).sum(1)     # 0 = correct
    stud_rank = (R["student"] > R["student"].gather(1, y[:, None])).sum(1)
    df = pd.DataFrame(dict(stream=sname, pos=np.arange(len(idx)), y=y.numpy(), domain=E.dom[idx],
                           final=R["final"].argmax(1).numpy(), final_rank=final_rank.numpy(),
                           student=R["student"].argmax(1).numpy(), student_rank=stud_rank.numpy(),
                           t3a=R["t3a"].argmax(1).numpy(), lame_only=R["lame_only"].argmax(1).numpy(),
                           nb_major=R["nb_major"].numpy(), flip=R["flip"].argmax(1).numpy(),
                           avg2=((R["student"].softmax(1) + R["flip"].softmax(1)) / 2).argmax(1).numpy()))
    rows.append(df)
    assert (df.final.values == pred).all()
    print("done", sname, f"final acc {(df.final == df.y).mean():.4f}", flush=True)
D = pd.concat(rows, ignore_index=True)
D.to_csv(E.ROOT / "analysis/error_records.csv", index=False)


def report(d, title):
    n = len(d)
    print(f"\n######## {title}  ({n} images)")
    print(f"final top-1 {(d.final_rank == 0).mean():.4f}  top-2 {(d.final_rank <= 1).mean():.4f}  "
          f"top-3 {(d.final_rank <= 2).mean():.4f}  top-5 {(d.final_rank <= 4).mean():.4f}")
    print(f"student top-1 {(d.student_rank == 0).mean():.4f}  top-3 {(d.student_rank <= 2).mean():.4f}")
    w = d[d.final != d.y]
    g1 = w[(w.final_rank == 1) & ((w.nb_major == w.y) | (w.t3a == w.y) | (w.lame_only == w.y))]
    same_wrong = (w.student == w.final) & (w.t3a == w.final) & (w.nb_major == w.final)
    g2 = w[(w.final_rank <= 2) & same_wrong & ~w.index.isin(g1.index)]
    g3 = w[w.final_rank > 2]
    rest = len(w) - len(g1) - len(g2) - len(g3)
    print(f"errors {len(w)} ({len(w) / n:.1%}):  G1 2nd + a signal supports truth {len(g1)}  |  "
          f"G2 top-3, all signals agree on the same wrong label {len(g2)}  |  G3 truth not in top-3 {len(g3)}  |  "
          f"other top-3 {rest}")
    per = len(d.stream.unique())
    print(f"  per val run: errors {len(w) / per:.0f}, G1 {len(g1) / per:.0f}, G2 {len(g2) / per:.0f}, "
          f"G3 {len(g3) / per:.0f}, other {rest / per:.0f}   (95% needs ~{(len(w) / per) - 0.05 * 2112:.0f} fewer errors)")
    print("  G3 by domain:", g3.domain.value_counts().to_dict(), " | errors by domain:", w.domain.value_counts().to_dict())
    print("  G3 by true class:", {CLS[k]: v for k, v in g3.y.value_counts().head(6).items()})
    print("  top confusions (true -> predicted):",
          {f"{CLS[a]}->{CLS[b]}": c for (a, b), c in w.groupby(["y", "final"]).size().sort_values(ascending=False).head(6).items()})
    # changes relative to the student
    ch = d[d.final != d.student]
    both = (d.t3a == d.lame_only) & (d.t3a != d.student)
    for name, sub in [("final != student (all changes)", ch), ("T3A and LAME-alone both support the same new label", d[both]),
                      ("only one of T3A / LAME-alone supports a change", d[(d.t3a != d.student) ^ (d.lame_only != d.student)])]:
        fix = ((sub.final == sub.y) & (sub.student != sub.y)).sum(); brk = ((sub.final != sub.y) & (sub.student == sub.y)).sum()
        print(f"  {name}: n={len(sub)}  wrong->right {fix}  right->wrong {brk}  wrong->wrong {len(sub) - fix - brk}")
    sub = d[both]
    print(f"    in the 'both support' subset the agreed label is correct {(sub.t3a == sub.y).mean():.1%} of the time")
    # flipped view
    sw, fr = d.student != d.y, d.flip == d.y
    print(f"  flip view: student wrong & flip right {(sw & fr).sum() / per:.0f} per run, student right & flip wrong "
          f"{(~sw & ~fr).sum() / per:.0f} per run; student acc {(~sw).mean():.4f}, flip acc {fr.mean():.4f}, "
          f"avg(orig, flip) acc {(d.avg2 == d.y).mean():.4f}")
    fw = d.final != d.y
    print(f"  final wrong & flip right {(fw & fr).sum() / per:.0f} per run (upper bound of what a perfect view selector adds "
          f"on top of the final: {(fw & fr).sum() / n:.2%})")


report(D[D.stream == "official"], "official val order")
report(D, "official + 6 orders pooled")
