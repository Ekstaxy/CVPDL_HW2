"""How to combine the prediction views (original, flip, center crops) — recorded once, combined offline.

  .venv/bin/python analysis/views_sim.py   -> analysis/views_sim.log, analysis/views_sim.csv

Extra views never change the adaptation (eval mode, no grad, no statistics update, no RNG), so per
(seed, stream) the logits / features of every view are recorded once and every combination rule is simulated
offline with its own T3A + LAME. Per image we also record the darkness of its four corners (mean pixel value of
four 12x12 corner patches, 0-255), a property of the image itself that a method may use.

Rules:
  uniform  : plain average of the chosen views
  gated    : crop views only for images whose corners are dark (corner mean < THR), else original + flip
  entropy  : per-image weights softmax(-H_v / tau) over the chosen views (no image-specific rule)
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
from tta.arbn_teacher import AdaptiveRBNTeacher  # noqa: E402
from tta.arbn_tent import lame  # noqa: E402
from tta.common import ema_update, entropy, symmetric_cross_entropy  # noqa: E402
from tta.t3a import T3A  # noqa: E402

def crop(x, r):
    s = int(round(x.shape[-1] * r)); o = (x.shape[-1] - s) // 2
    return F.interpolate(x[:, :, o:o + s, o:o + s], size=x.shape[-2:], mode="bilinear", align_corners=False)


VIEWS = {"F": lambda x: torch.flip(x, dims=[3])}
for r in (0.8, 0.7, 0.6):
    VIEWS[f"C{int(r * 100)}"] = lambda x, r=r: crop(x, r)
    VIEWS[f"CF{int(r * 100)}"] = lambda x, r=r: torch.flip(crop(x, r), dims=[3])
CACHE = E.ROOT / "out" / "_cache" / "views"
CACHE.mkdir(parents=True, exist_ok=True)


class Rec(AdaptiveRBNTeacher):
    def __init__(self, model, **kw):
        super().__init__(model, flip=False, **kw)
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
            r = {"O": (s_out.detach().half().cpu(), self._feats.detach().half().cpu())}
            with torch.no_grad():
                self.model.eval()
                for v, fn in VIEWS.items():
                    o = self.model(fn(x)); r[v] = (o.half().cpu(), self._feats.detach().half().cpu())
                self.model.train()
                raw = x * E.STD + E.MEAN
                c = 12
                corners = torch.stack([raw[:, :, :c, :c], raw[:, :, :c, -c:], raw[:, :, -c:, :c], raw[:, :, -c:, -c:]])
                r["corner"] = (corners.mean(dim=(0, 2, 3, 4)) * 255).cpu()
            self.optimizer.zero_grad()
            (self.w_ent * entropy(s_out).mean()).backward()
            self.model.eval()
            a_out = self.model(self.transform(x))
            self.model.train()
            (self.w_cons * symmetric_cross_entropy(a_out, t_out).mean()).backward()
            self.optimizer.step()
        ema_update(self.teacher, self.model, self.mt)
        self.rec.append(r)
        return s_out.argmax(1)


def record(seed, sname, idx):
    f = CACHE / f"{seed}_{sname}.pt"
    if not f.exists():
        _, m = E.stream(lambda mm: Rec(mm), idx, seed=seed)
        torch.save(m.rec, f)
    return torch.load(f)


def combine(batch, views, rule, thr=None, tau=None, crop_views=()):
    L = torch.stack([batch[v][0].float() for v in views])     # (V, B, C)
    Fe = torch.stack([batch[v][1].float() for v in views])
    V = len(views)
    if rule == "uniform":
        w = torch.full((V, L.shape[1]), 1.0 / V)
    elif rule == "gated":
        dark = batch["corner"] < thr                            # (B,)
        is_crop = torch.tensor([v in crop_views for v in views])
        w = torch.where(is_crop[:, None], dark[None].float(), torch.ones(V, L.shape[1]))
        w = w / w.sum(0, keepdim=True)
    elif rule == "entropy":
        H = -(L.softmax(-1) * L.log_softmax(-1)).sum(-1)       # (V, B)
        w = (-H / tau).softmax(0)
    return (w[..., None] * L).sum(0), (w[..., None] * Fe).sum(0)


FC = E.BASE.fc
streams = list(E.ORDERS.items())
RULES = [("OF uniform", ["O", "F"], "uniform", {}),
         ("OF+C70+CF70 uniform", ["O", "F", "C70", "CF70"], "uniform", {}),
         ("OF+C60+CF60 uniform", ["O", "F", "C60", "CF60"], "uniform", {}),
         ("OF+C80+CF80+C70+CF70 uniform", ["O", "F", "C80", "CF80", "C70", "CF70"], "uniform", {})]
for vs, thr in itertools.product([("C80", "CF80"), ("C70", "CF70"), ("C60", "CF60"), ("C70", "CF70", "C80", "CF80")],
                                 [5, 15, 30]):
    RULES.append((f"OF+{'+'.join(vs)} gated corner<{thr}", ["O", "F", *vs], "gated", dict(thr=thr, crop_views=vs)))
for vs, tau in itertools.product([("C70", "CF70"), ("C80", "CF80"), ("C80", "CF80", "C70", "CF70")], [0.25, 0.5, 1.0]):
    RULES.append((f"OF+{'+'.join(vs)} entropy tau={tau}", ["O", "F", *vs], "entropy", dict(tau=tau)))

rows = []
for seed in (0, 1, 2):
    for sname, idx in streams:
        rec = record(seed, sname, idx)
        y = E.y[idx]
        for name, views, rule, kw in RULES:
            t3a = T3A(FC, 50, 20)
            preds = []
            for b in rec:
                lg, ft = combine(b, views, rule, **kw)
                lg, ft = lg.cuda(), ft.cuda()
                preds.append(lame(t3a(ft, lg), ft, 5).argmax(1).cpu())
            ok = torch.cat(preds).numpy() == y
            rows.append(dict(rule=name, seed=seed, stream=sname, acc=ok.mean(),
                             **{d: ok[E.dom[idx] == d].mean() for d in E.DOMS}))
        print("done", seed, sname, flush=True)
df = pd.DataFrame(rows)
df.to_csv(E.ROOT / "analysis/views_sim.csv", index=False)

# dark-corner statistics per domain (is the gate selective?)
cs = []
for seed, (sname, idx) in [(0, streams[0])]:
    c = torch.cat([b["corner"] for b in record(seed, sname, idx)]).numpy()
    for d in E.DOMS:
        v = c[E.dom[idx] == d]
        cs.append(f"{d}: corner mean p5 {np.percentile(v, 5):.1f} / median {np.median(v):.1f} / p95 {np.percentile(v, 95):.1f}"
                  f" | <5: {(v < 5).mean():.1%} <15: {(v < 15).mean():.1%} <30: {(v < 30).mean():.1%}")
print("\n".join(cs))

base = df[df.rule == "OF uniform"].set_index(["seed", "stream"]).acc
out = []
for name, g in df.groupby("rule", sort=False):
    a = g.set_index(["seed", "stream"]).acc
    per_seed = g[g.stream != "official"].groupby("seed").acc
    diff = a - base
    out.append(dict(rule=name, official=g[g.stream == "official"].acc.mean(), orders_mean=per_seed.mean().mean(),
                    orders_worst=per_seed.min().mean(), **{d: g[d].mean() for d in E.DOMS},
                    vs_OF_mean=diff.mean(), vs_OF_min=diff.min(), wins=f"{(diff > 0).sum()}/{len(diff)}"))
pd.set_option("display.width", 250)
print(pd.DataFrame(out).sort_values("orders_worst", ascending=False).round(4).to_string(index=False))
