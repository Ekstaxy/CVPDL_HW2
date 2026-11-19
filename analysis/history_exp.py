"""History experiments for arbn_teacher (A = current default, B = clear SGD momentum at a detected shift).

  .venv/bin/python analysis/history_exp.py   -> analysis/history_exp.log, analysis/history_records.csv

Aligned streams: each val domain is cut to a multiple of 64 (scanline 768, wave 704, fisheye 576 images, first
images in the official within-domain order), so domain boundaries fall on batch boundaries and, for a given image,
the 63 batch companions are identical in all 6 domain orders. Only the history before the image differs.

Recorded per image (labels only used afterwards): correct after (a) original+flip student, (b) T3A, (c) LAME =
final; majority vote of its 5 nearest neighbours (cosine, prediction features) within the current batch and among
all earlier batches of the stream (using their final predicted labels). Per batch: cosine between the SGD momentum
buffer and the fresh gradient right before optimizer.step().
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
import tta.arbn_teacher as AT  # noqa: E402

LEN = {"scanline": 768, "wave": 704, "fisheye": 576}
DOM_IDX = {d: np.where(E.dom == d)[0][:LEN[d]] for d in E.DOMS}
ALIGNED = {"-".join(p): (np.concatenate([DOM_IDX[d] for d in p]), p) for p in itertools.permutations(E.DOMS)}

_orig_lame = AT.lame


class Rec(AT.AdaptiveRBNTeacher):
    def __init__(self, model, **kw):
        super().__init__(model, **kw)
        self.r = []                      # per batch dict
        inner_t3a = self.t3a

        def t3a_spy(feats, logits):
            out = inner_t3a(feats, logits)
            self._cur = dict(base=logits.argmax(1).cpu(), t3a=out.argmax(1).cpu(), feats=feats.half().cpu())
            return out
        self.t3a = t3a_spy
        inner_step = self.optimizer.step

        def step_spy(*a, **k):
            m, g = [], []
            for grp in self.optimizer.param_groups:
                for p in grp["params"]:
                    buf = self.optimizer.state[p].get("momentum_buffer")
                    if buf is not None and p.grad is not None:
                        m.append(buf.flatten()); g.append(p.grad.flatten())
            self._cur["mom_cos"] = float(F.cosine_similarity(torch.cat(m), torch.cat(g), dim=0)) if m else np.nan
            return inner_step(*a, **k)
        self.optimizer.step = step_spy

    def __call__(self, x):
        pred = super().__call__(x)
        self._cur.update(final=pred.cpu(), alpha=self.alpha_hist[-1])
        self.r.append(self._cur)
        return pred


def run(variant, idx, seed):
    kw = {} if variant == "A" else dict(mom_reset_alpha=0.3)
    _, m = E.stream(lambda mm: Rec(mm, **kw), idx, seed=seed)
    return m.r


def neighbour_votes(r, k=5):
    feats = torch.cat([b["feats"] for b in r]).float().cuda()
    final = torch.cat([b["final"] for b in r]).cuda()
    f = F.normalize(feats, dim=1)
    cur, hist = [], []
    for b in range(len(r)):
        s = slice(b * 64, (b + 1) * 64)
        sim = f[s] @ f[s].T
        sim.fill_diagonal_(-2)
        cur.append(torch.mode(final[s][sim.topk(k, dim=1).indices], dim=1).values)
        if b == 0:
            hist.append(torch.full((64,), -1, device=f.device))
        else:
            h = f[s] @ f[: b * 64].T
            hist.append(torch.mode(final[: b * 64][h.topk(k, dim=1).indices], dim=1).values)
    return torch.cat(cur).cpu().numpy(), torch.cat(hist).cpu().numpy()


rows, batches = [], []
for variant in ("A", "B"):
    for seed in (0, 1, 2):
        for oname, (idx, perm) in ALIGNED.items():
            r = run(variant, idx, seed)
            y = E.y[idx]; d = E.dom[idx]
            seg = np.concatenate([[i] * LEN[p] for i, p in enumerate(perm)])
            prev = np.concatenate([["none" if i == 0 else perm[i - 1]] * LEN[p] for i, p in enumerate(perm)])
            starts = np.concatenate([[sum(LEN[q] for q in perm[:i])] * LEN[p] for i, p in enumerate(perm)])
            pos = np.arange(len(idx))
            cur_v, hist_v = neighbour_votes(r) if variant == "A" else (np.full(len(idx), -1), np.full(len(idx), -1))
            cat = lambda k: torch.cat([b[k] for b in r]).numpy()
            rows.append(pd.DataFrame(dict(variant=variant, seed=seed, order=oname, img=idx, y=y, domain=d, seg=seg,
                                          prev=prev, bidx=(pos - starts) // 64, ok_base=cat("base") == y,
                                          ok_t3a=cat("t3a") == y, ok_final=cat("final") == y,
                                          cur_vote=cur_v, hist_vote=hist_v)))
            for b, rb in enumerate(r):
                batches.append(dict(variant=variant, seed=seed, order=oname, batch=b, alpha=rb["alpha"],
                                    mom_cos=rb["mom_cos"], domain=d[b * 64], bidx=(b * 64 - starts[b * 64]) // 64,
                                    seg=seg[b * 64]))
        print(variant, seed, "done", flush=True)
D = pd.concat(rows, ignore_index=True)
Bt = pd.DataFrame(batches)
D.to_csv(E.ROOT / "analysis/history_records.csv", index=False)
Bt.to_csv(E.ROOT / "analysis/history_batches.csv", index=False)
print("saved records")
