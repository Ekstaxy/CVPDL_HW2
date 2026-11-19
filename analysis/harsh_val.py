"""Harsher labelled proxy stream (evaluation only): the val stream with a second, synthetic corruption on top, changing
every 352 images (6 segments; noise in low light, posterize, false colour, stripes, pixelate, dither). The real test
stream has much larger shifts than val; this proxy is meant to rank configurations under large shifts without
spending Kaggle submissions.

  .venv/bin/python analysis/harsh_val.py <base config json> '<json list of {name, cfg}>'
"""
import json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as E  # noqa: E402
from tta.arbn_teacher import AdaptiveRBNTeacher  # noqa: E402


import os
MILD = os.environ.get("PROXY", "harsh") == "mild"   # PROXY=mild: weaker second corruption (accuracy near the test level)


def corrupt(x, kind, g):
    x = x.float()
    if MILD:
        if kind == "lownoise":
            x = x * 0.7 + torch.randn(x.shape, generator=g) * 12
        elif kind == "posterize":
            x = torch.floor(x / 48) * 48 + 24
        elif kind == "falsecolor":   # channel rotation + inverted green
            x = torch.stack([x[:, 2], 255 - x[:, 1], x[:, 0]], 1)
        elif kind == "stripes":
            i = torch.arange(x.shape[-1]).float()
            x = x * (0.85 + 0.15 * torch.sign(torch.sin(2 * np.pi * (i[:, None] + i[None, :]) / 9)))
        elif kind == "pixelate":
            x = F.interpolate(F.avg_pool2d(x, 2), size=x.shape[-2:], mode="nearest")
        elif kind == "dither":
            b = torch.tensor([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]).float() / 16 - 0.5
            t = b.repeat(x.shape[-2] // 4, x.shape[-1] // 4)
            x = torch.floor((x / 255 * 7 + t + 0.5).clamp(0, 7)) / 7 * 255
        return x.clamp(0, 255).round().to(torch.uint8)
    if kind == "lownoise":
        x = x * 0.45 + torch.randn(x.shape, generator=g) * 28
    elif kind == "posterize":
        x = torch.floor(x / 64) * 64 + 32
    elif kind == "falsecolor":
        l = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]) / 255
        x = torch.stack([(1.5 - (4 * l - 3).abs()), (1.5 - (4 * l - 2).abs()), (1.5 - (4 * l - 1).abs())], 1).clamp(0, 1) * 255
    elif kind == "stripes":
        i = torch.arange(x.shape[-1]).float()
        m = 0.65 + 0.35 * torch.sign(torch.sin(2 * np.pi * (i[:, None] + i[None, :]) / 9))
        x = x * m
    elif kind == "pixelate":
        x = F.interpolate(F.avg_pool2d(x, 5), size=x.shape[-2:], mode="nearest")
    elif kind == "dither":
        b = torch.tensor([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]).float() / 16 - 0.5
        t = b.repeat(x.shape[-2] // 4, x.shape[-1] // 4)
        x = torch.floor((x / 255 * 3 + t + 0.5).clamp(0, 3)) / 3 * 255
    return x.clamp(0, 255).round().to(torch.uint8)


KINDS = ["lownoise", "posterize", "falsecolor", "stripes", "pixelate", "dither"]
SEG = 352
X0 = E.X
STREAMS = []
for shift in (0, 3):
    g = torch.Generator().manual_seed(shift)
    X = X0.clone(); seg = np.zeros(len(X0), int)
    for k, s in enumerate(range(0, len(X0), SEG)):
        X[s:s + SEG] = corrupt(X0[s:s + SEG], KINDS[(k + shift) % 6], g); seg[s:s + SEG] = (k + shift) % 6
    STREAMS.append((X, seg))

if __name__ == "__main__":
    BASE_CFG = json.load(open(sys.argv[1]))
    jobs = [{"name": "base", "cfg": {}}] + json.loads(sys.argv[2])
    idx = E.ORDERS["official"]; base = None
    for j in jobs:
        c = {**BASE_CFG, **j["cfg"]}; c["upscales"] = tuple(c.get("upscales", ()))
        acc, per = [], np.zeros((6, 2))
        for seed in (0, 1, 2):
            for X, seg in STREAMS:
                E.X = X
                pred, _ = E.stream(lambda m: AdaptiveRBNTeacher(m, **c), idx, seed=seed)
                ok = pred == E.y; acc.append(ok.mean())
                for k in range(6):
                    per[k] += [ok[seg == k].sum(), (seg == k).sum()]
        E.X = X0
        a = np.array(acc); base = a if base is None else base
        print(f"{j['name']:<36} harsh {a.mean():.4f}  paired {100 * (a - base).mean():+.2f} (min {100 * (a - base).min():+.2f}, wins {(a > base).sum()}/6)  "
              + " ".join(f"{KINDS[k][:5]} {per[k, 0] / per[k, 1]:.3f}" for k in range(6)), flush=True)
