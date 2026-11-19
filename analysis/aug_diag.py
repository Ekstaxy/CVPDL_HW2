"""Which part of the strong augmentation hides the cues? Diagnostic views on the current default (A).

  .venv/bin/python analysis/aug_diag.py   -> analysis/aug_diag.log, analysis/aug_diag_records.csv

On the aligned val streams (see history_exp.py), for every batch the student is additionally evaluated on:
  orig, color only (ColorJitterPro), geometric only (pad + affine + center crop), blur + noise only, full strong aug.
The views are computed with the same model state as the answer (after the BN statistics update, before the SGD
step), in eval mode (no statistics written), without gradients, outside T3A, and with the CPU / CUDA RNG state saved
and restored, so the adaptation trajectory and the answers are exactly those of the default method.
Augmentation parameters are drawn once per batch, as in training.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as E  # noqa: E402
from tta.arbn_teacher import AdaptiveRBNTeacher  # noqa: E402
from tta.common import Clip, ColorJitterPro, GaussianNoise, NormalizedAug, get_tta_transforms  # noqa: E402

LEN = {"scanline": 768, "wave": 704, "fisheye": 576}   # same aligned streams as history_exp.py
DOM_IDX = {d: np.where(E.dom == d)[0][:LEN[d]] for d in E.DOMS}
ALIGNED = {"-".join(p): (np.concatenate([DOM_IDX[d] for d in p]), p) for p in itertools.permutations(E.DOMS)}

S = 160
VIEWS = {
    "color": NormalizedAug(T.Compose([Clip(), ColorJitterPro([0.6, 1.4], [0.7, 1.3], [0.5, 1.5], [-0.06, 0.06],
                                                             [0.7, 1.3]), Clip()])),
    "geometric": NormalizedAug(T.Compose([T.Pad(S // 2, padding_mode="edge"),
                                          T.RandomAffine(degrees=[-15, 15], translate=(1 / 16, 1 / 16), scale=(0.9, 1.1),
                                                         interpolation=T.InterpolationMode.BILINEAR, fill=0),
                                          T.CenterCrop(S)])),
    "blur+noise": NormalizedAug(T.Compose([T.GaussianBlur(5, sigma=[0.001, 0.5]), GaussianNoise(0.005), Clip()])),
    "full": NormalizedAug(get_tta_transforms()),
}


class Diag(AdaptiveRBNTeacher):
    def __init__(self, model, **kw):
        super().__init__(model, **kw)
        self.r = []
        inner = self.t3a

        def spy(feats, logits):
            out = inner(feats, logits)
            with torch.no_grad():
                cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
                was_training = self.model.training
                self.model.eval()
                rec = {"base": logits.argmax(1).cpu(), "t3a": out.argmax(1).cpu(),
                       "orig": self.model(self._x).argmax(1).cpu()}
                for k, tf in VIEWS.items():
                    rec[k] = self.model(tf(self._x)).argmax(1).cpu()
                self.model.train(was_training)
                torch.set_rng_state(cpu); torch.cuda.set_rng_state(cuda)
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
            pred, m = E.stream(lambda mm: Diag(mm), idx, seed=seed)
            y = E.y[idx]
            starts = np.concatenate([[sum(LEN[q] for q in perm[:i])] * LEN[p] for i, p in enumerate(perm)])
            cat = lambda k: torch.cat([b[k] for b in m.r]).numpy()
            rows.append(pd.DataFrame(dict(seed=seed, order=oname, img=idx, y=y, domain=E.dom[idx],
                                          bidx=(np.arange(len(idx)) - starts) // 64,
                                          **{f"ok_{k}": cat(k) == y for k in ["base", "t3a", "final", "orig", *VIEWS]})))
        print("seed", seed, "done", flush=True)
    D = pd.concat(rows, ignore_index=True)
    D.to_csv(E.ROOT / "analysis/aug_diag_records.csv", index=False)
    RUNS = 18
    www = D[(D.bidx >= 2) & ~D.ok_base & ~D.ok_t3a & ~D.ok_final]
    print(f"\nstudent accuracy on the original view: {D.ok_orig.mean():.4f}; hard steady WWW images: {len(www) / RUNS:.0f}/run")
    print(f"{'view':12s} {'acc':>7s} {'right->wrong (of orig right)':>30s} {'wrong->right (of orig wrong)':>30s} {'WWW rescued':>12s}")
    for v in ["orig", *VIEWS]:
        r, w = D[D.ok_orig], D[~D.ok_orig]
        print(f"{v:12s} {D[f'ok_{v}'].mean():7.4f} {(~r[f'ok_{v}']).mean():29.1%} {w[f'ok_{v}'].mean():30.1%} "
              f"{www[f'ok_{v}'].mean():11.1%}")
    print("\nright->wrong rate by domain:")
    for v in VIEWS:
        r = D[D.ok_orig]
        print(f"  {v:12s} " + "  ".join(f"{d} {(~r[r.domain == d][f'ok_{v}']).mean():.1%}" for d in E.DOMS))
    print("WWW rescued by domain:")
    for v in VIEWS:
        print(f"  {v:12s} " + "  ".join(f"{d} {www[www.domain == d][f'ok_{v}'].mean():.1%} (n={len(www[www.domain == d]) / RUNS:.0f})"
                                        for d in E.DOMS))
