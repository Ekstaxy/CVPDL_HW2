"""Local pooling of the last feature map as an extra candidate answer (diagnostic only).

  .venv/bin/python analysis/local_feat_diag.py   -> analysis/local_feat_diag.log, analysis/local_feat_records.csv

At answer time (same model state as the student's answer: after the BN statistics update, before the SGD step),
the layer4 feature map (2048 x 5 x 5 for a 160 x 160 input) of the original and the flipped image is read in eval
mode without gradients (no statistics written, no randomness, outside T3A). The flipped map is flipped back.
Candidates = the source classifier (fc) applied to different spatial poolings, averaged over the two views:
  global (= the usual average pool), center 3x3, center-weighted 2:1, center-weighted 4:1.
Aligned val streams (see history_exp.py), 3 seeds. Answers and adaptation are exactly the default method's.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as E  # noqa: E402
from tta.arbn_teacher import AdaptiveRBNTeacher  # noqa: E402

LEN = {"scanline": 768, "wave": 704, "fisheye": 576}
DOM_IDX = {d: np.where(E.dom == d)[0][:LEN[d]] for d in E.DOMS}
ALIGNED = {"-".join(p): (np.concatenate([DOM_IDX[d] for d in p]), p) for p in itertools.permutations(E.DOMS)}


def pool_weights(h, w, device):
    center = torch.zeros(h, w, device=device)
    center[1:h - 1, 1:w - 1] = 1
    out = {"global": torch.ones(h, w, device=device), "center3x3": center.clone()}
    for k in (2, 4):
        out[f"weighted{k}:1"] = 1 + (k - 1) * center
    return {n: v / v.sum() for n, v in out.items()}


class Diag(AdaptiveRBNTeacher):
    def __init__(self, model, **kw):
        super().__init__(model, **kw)
        self.r = []
        self._fmap = None
        self.model.layer4.register_forward_hook(lambda m, i, o: setattr(self, "_fmap", o))
        inner = self.t3a

        def spy(feats, logits):
            out = inner(feats, logits)
            with torch.no_grad():
                was = self.model.training
                self.model.eval()
                self.model(self._x); f0 = self._fmap.float()
                self.model(torch.flip(self._x, dims=[3])); f1 = torch.flip(self._fmap.float(), dims=[3])
                self.model.train(was)
                W = pool_weights(f0.shape[2], f0.shape[3], f0.device)
                rec = {"base": logits.argmax(1).cpu(), "t3a": out.argmax(1).cpu()}
                for n, w in W.items():
                    z0 = (f0 * w).sum((2, 3)); z1 = (f1 * w).sum((2, 3))
                    rec[n] = ((self.model.fc(z0) + self.model.fc(z1)) / 2).argmax(1).cpu()
            self._rec = rec
            return out
        self.t3a = spy

    def __call__(self, x):
        self._x = x
        pred = super().__call__(x)
        self._rec["final"] = pred.cpu()
        self.r.append(self._rec)
        return pred


if __name__ == "__main__":
    rows = []
    for seed in (0, 1, 2):
        for oname, (idx, perm) in ALIGNED.items():
            _, m = E.stream(lambda mm: Diag(mm), idx, seed=seed)
            y = E.y[idx]
            starts = np.concatenate([[sum(LEN[q] for q in perm[:i])] * LEN[p] for i, p in enumerate(perm)])
            cat = lambda k: torch.cat([b[k] for b in m.r]).numpy()
            keys = [k for k in m.r[0]]
            rows.append(pd.DataFrame(dict(seed=seed, order=oname, img=idx, y=y, domain=E.dom[idx],
                                          bidx=(np.arange(len(idx)) - starts) // 64,
                                          **{f"ok_{k}": cat(k) == y for k in keys})))
        print("seed", seed, "done", flush=True)
    D = pd.concat(rows, ignore_index=True)
    D.to_csv(E.ROOT / "analysis/local_feat_records.csv", index=False)
    RUNS = 18
    CANDS = ["global", "center3x3", "weighted2:1", "weighted4:1"]
    print(f"\nsanity: global-pool candidate agrees with the student's original+flip answer on "
          f"{(D.ok_global == D.ok_base).mean():.1%} of images (correctness)")
    www = D[(D.bidx >= 2) & ~D.ok_base & ~D.ok_t3a & ~D.ok_final]
    fr, fw = D[D.ok_final], D[~D.ok_final]
    print(f"final accuracy {D.ok_final.mean():.4f}; hard WWW (steady, all stages wrong): {len(www) / RUNS:.0f}/run\n")
    print(f"{'candidate':12s} {'acc':>7s} {'WWW -> right':>13s} {'final wrong -> right':>21s} {'final right -> wrong':>21s} "
          f"{'net if it replaced final':>25s}")
    for c in CANDS:
        k = f"ok_{c}"
        net = (fw[k].sum() - (~fr[k]).sum()) / RUNS
        print(f"{c:12s} {D[k].mean():7.4f} {www[k].mean():12.1%} {fw[k].sum() / RUNS:16.1f}/run {(~fr[k]).sum() / RUNS:16.1f}/run {net:20.1f}/run")
    print("\nby domain (WWW rescued % | final-right broken %):")
    for c in CANDS[1:]:
        k = f"ok_{c}"
        print(f"  {c:12s} " + "  ".join(
            f"{d}: {www[www.domain == d][k].mean():.1%} | {(~fr[fr.domain == d][k]).mean():.1%}" for d in E.DOMS))
    print("\nstudent-level comparison (vs global pool, same views): of images global gets wrong, local gets right / "
          "of images global gets right, local gets wrong")
    for c in CANDS[1:]:
        k = f"ok_{c}"
        g_w, g_r = D[~D.ok_global], D[D.ok_global]
        print(f"  {c:12s} {g_w[k].mean():.1%} / {(~g_r[k]).mean():.1%}")
