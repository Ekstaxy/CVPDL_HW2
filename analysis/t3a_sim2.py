"""T3A variants replayed on the cached view records (final stage = LAME k=5 + pow + 0.1*box(8)).

  A  current: supports pseudo-labelled by the raw (view-averaged) logits, current batch joins before scoring
  B  refined commit: score as A, but the supports kept for later batches carry the FINAL label / entropy (after LAME)
  C  mix: T3A logits + w * linear-classifier logits as the unary term
"""
import sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
import evalkit as E  # noqa: E402
from tta.arbn_tent import lame  # noqa: E402

CACHE = E.ROOT / "out/_cache/auto_views"; VIEWS = ["O", "F", "U224", "UF224"]
LK = dict(power=True, time_w=0.1, time_tau=8)

class T3Av:
    def __init__(self, fc, k=50, c=20):
        self.k, self.c = k, c
        w = fc.weight.detach().float().clone(); p0 = (w @ w.T + fc.bias.detach().float()).softmax(1)
        self.s, self.y, self.e = w, p0.argmax(1), -(p0 * p0.clamp_min(1e-12).log()).sum(1)
    def select(self, S, Y, Ent):
        keep = torch.zeros(len(S), dtype=torch.bool, device=S.device)
        for c in range(self.c):
            ic = (Y == c).nonzero().flatten()
            if len(ic): keep[ic[Ent[ic].argsort()[: self.k]]] = True
        return S[keep], Y[keep], Ent[keep]
    def protos(self, S, Y):
        return F.normalize(F.one_hot(Y, self.c).float().T @ F.normalize(S, dim=1), dim=1)
    def score(self, z, p):
        ent = -(p * p.clamp_min(1e-12).log()).sum(1)
        S, Y, En = self.select(torch.cat([self.s, z]), torch.cat([self.y, p.argmax(1)]), torch.cat([self.e, ent]))
        return z @ self.protos(S, Y).T, (S, Y, En)
    def commit(self, state=None, z=None, p=None):
        if state is not None: self.s, self.y, self.e = state
        else:
            ent = -(p * p.clamp_min(1e-12).log()).sum(1)
            self.s, self.y, self.e = self.select(torch.cat([self.s, z]), torch.cat([self.y, p.argmax(1)]), torch.cat([self.e, ent]))

def run(mode, k=50, mixw=0.0, ent_src="final"):
    accs, off, sw, fi = [], [], [], []
    for seed in (0, 1, 2):
        for s, idx in E.ORDERS.items():
            rec = torch.load(CACHE / f"{seed}_{s}.pt", weights_only=False)
            t = T3Av(E.BASE.fc, k); P = []
            for b in rec:
                lg = torch.stack([b[v][0].float() for v in VIEWS]).mean(0).cuda(); ft = torch.stack([b[v][1].float() for v in VIEWS]).mean(0).cuda()
                p = lg.softmax(1)
                tl, st = t.score(ft, p)
                Z = lame(tl + mixw * lg, ft, 5, **LK)
                if mode == "A": t.commit(state=st)
                else: t.commit(z=ft, p=Z if ent_src == "final" else F.one_hot(Z.argmax(1), 20).float() * 0 + p * F.one_hot(Z.argmax(1), 20) + 1e-6)
                P.append(Z.argmax(1).cpu())
            ok = torch.cat(P).numpy() == E.y[idx]; d = E.dom[idx]
            accs.append(ok.mean()); sw.append(ok[d != "fisheye"].mean()); fi.append(ok[d == "fisheye"].mean())
            if s == "official": off.append(ok.mean())
    return np.array(accs), np.mean(off), np.mean(sw), np.mean(fi)

base = None
for name, kw in [("A current T3A", dict(mode="A")), ("B refined commit (final label + final entropy)", dict(mode="B")),
                 ("B refined commit, K=20", dict(mode="B", k=20)), ("B refined commit, K=100", dict(mode="B", k=100)),
                 ("A + 0.5 * fc logits", dict(mode="A", mixw=0.5)), ("A + 1.0 * fc logits", dict(mode="A", mixw=1.0)),
                 ("A + 0.25 * fc logits", dict(mode="A", mixw=0.25))]:
    a, o, s_, f_ = run(**kw)
    if base is None: base = a
    print(f"{name:<50} all {a.mean():.4f}  official {o:.4f}  scan+wave {s_:.4f}  fisheye {f_:.4f}  paired {100*(a-base).mean():+.2f} (min {100*(a-base).min():+.2f}, wins {(a>base).sum()}/21)", flush=True)
