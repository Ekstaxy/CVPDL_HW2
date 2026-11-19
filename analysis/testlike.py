"""Test-like labelled proxy (evaluation only): the long val stream (first 192 batches = 12,288 images, the test length)
cut into the 12 segment lengths of the test stream (batches 18,18,10,16,16,14,13,13,20,18,19,17), each segment with a
synthetic copy of the corruption seen in the matching test segment (identified by eye from test images):

  0 lownoise  1 swirl  2 chroma  3 inferno  4 sepia  5 jpeg  6 diag  7 halftone  8 teal  9 dotgrid  10 dither  11 oil

The val images already carry their own corruption (scanline / wave / fisheye), so this is harder than the test.
Use it to rank configurations, per segment, without Kaggle submissions.

  .venv/bin/python analysis/testlike.py <base config json> '<json list of {name, cfg}>' [seeds]
  .venv/bin/python analysis/testlike.py --show       # writes a montage to compare with the real test segments
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as E  # noqa: E402

SEG_B = [18, 18, 10, 16, 16, 14, 13, 13, 20, 18, 19, 17]
KINDS = ["lownoise", "swirl", "chroma", "inferno", "sepia", "jpeg", "diag", "halftone", "teal", "dotgrid", "dither", "oil"]


def _pil_map(x, fn):
    out = []
    for im in x:
        out.append(torch.from_numpy(np.asarray(fn(Image.fromarray(im.permute(1, 2, 0).numpy())).convert("RGB")).copy()))
    return torch.stack(out).permute(0, 3, 1, 2).float()


def _jpeg(q):
    def f(im):
        b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0)
        return Image.open(b)
    return f


def _blur(x, s):
    k = int(2 * round(3 * s) + 1); t = torch.arange(k).float() - k // 2
    g = torch.exp(-t ** 2 / (2 * s * s)); g = g / g.sum()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)


import os
MILD = os.environ.get("PROXY", "mild") == "mild"   # PROXY=harsh: the stronger first version


def corrupt(x, kind, g):
    """x: uint8 (N,3,H,W) -> uint8."""
    m = MILD
    n, _, h, w = x.shape
    xf = x.float()
    yy, xx = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    if kind == "lownoise":   # dark + strong colour noise
        out = xf * (0.7 if m else 0.55) + torch.randn(xf.shape, generator=g) * (22 if m else 38)
    elif kind == "swirl":    # strong horizontal wave warp with two frequencies + slight vertical
        ph = torch.rand(n, 2, generator=g) * 6.28
        dx = (7 if m else 14) * torch.sin(2 * np.pi * yy / 23 + ph[:, :1, None]) + (4 if m else 8) * torch.sin(2 * np.pi * yy / 9 + ph[:, 1:, None])
        dy = (2 if m else 4) * torch.sin(2 * np.pi * xx / 31 + ph[:, :1, None])
        gx = (xx + dx) / (w - 1) * 2 - 1; gy = (yy + dy) / (h - 1) * 2 - 1
        out = F.grid_sample(xf, torch.stack([gx, gy], -1), mode="bilinear", padding_mode="reflection", align_corners=True)
    elif kind == "chroma":   # radial chromatic aberration + blur + rainbow fringe
        b = _blur(xf, 1.2 if m else 2.0)
        cx, cy = (w - 1) / 2, (h - 1) / 2
        ch = []
        for c, s in zip(range(3), ((1.03, 1.0, 0.97) if m else (1.06, 1.0, 0.94))):
            gx = ((xx - cx) / s + cx) / (w - 1) * 2 - 1; gy = ((yy - cy) / s + cy) / (h - 1) * 2 - 1
            grid = torch.stack([gx, gy], -1).expand(n, h, w, 2)
            ch.append(F.grid_sample(b[:, c:c + 1], grid, mode="bilinear", padding_mode="border", align_corners=True))
        out = torch.cat(ch, 1) * 0.85 + 30
    elif kind == "inferno":  # luminance through the inferno colour map
        import matplotlib
        lut = torch.from_numpy(matplotlib.colormaps["inferno"](np.linspace(0, 1, 256))[:, :3]).float() * 255
        lum = (0.299 * xf[:, 0] + 0.587 * xf[:, 1] + 0.114 * xf[:, 2]).clamp(0, 255).long()
        out = lut[lum].permute(0, 3, 1, 2)
    elif kind == "sepia":    # sepia tone + grain + vertical scratches
        sm = torch.tensor([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
        out = torch.einsum("ij,njhw->nihw", sm, xf) * 0.8 + 15 + torch.randn(n, 1, h, w, generator=g) * (6 if m else 10)
        for i in range(n):
            for _ in range(int(torch.randint(2, 5, (1,), generator=g) if m else torch.randint(3, 8, (1,), generator=g))):
                c = int(torch.randint(0, w, (1,), generator=g))
                out[i, :, :, c] = out[i, :, :, c] * 0.4 + 150
    elif kind == "jpeg":     # haze + colour noise + very low JPEG quality
        t = xf * (0.85 if m else 0.75) + (30 if m else 50) + torch.randn(xf.shape, generator=g) * (6 if m else 12)
        out = _pil_map(t.clamp(0, 255).round().to(torch.uint8), _jpeg(12 if m else 6))
    elif kind == "diag":     # semi-transparent bright diagonal bands
        band = ((xx + yy) % 14 < 6).float()
        a = 0.3 if m else 0.45
        out = xf * (1 - a * band) + 255 * a * band
    elif kind == "halftone":  # brightness-modulated dot screen
        cell = 4
        lum = (0.299 * xf[:, 0] + 0.587 * xf[:, 1] + 0.114 * xf[:, 2]) / 255
        r = torch.sqrt((xx % cell - cell / 2 + 0.5) ** 2 + (yy % cell - cell / 2 + 0.5) ** 2) / (cell / 2)
        dot = (r < (1 - lum[:, None]) * 1.2).float()
        out = xf * (1 - (0.4 if m else 0.55) * dot)
    elif kind == "teal":     # underwater teal tint, low contrast, blur
        tint = torch.tensor([20.0, 120.0, 140.0]).view(1, 3, 1, 1)
        out = _blur(xf, 0.8) * 0.6 + tint * 0.4 if m else _blur(xf, 1.2) * 0.45 + tint * 0.55
    elif kind == "dotgrid":  # regular grid of dark dots
        dot = ((xx % 5 < 2) & (yy % 5 < 2)).float()
        out = xf * (1 - (0.35 if m else 0.5) * dot)
    elif kind == "dither":   # ordered dither to 2 levels per channel (+ noise)
        b = torch.tensor([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]).float() / 16 - 0.5
        t = b.repeat(h // 4, w // 4)
        out = (lambda L: torch.floor((xf / 255 * L + t + 0.5).clamp(0, L)) / L * 255)(5 if m else 3)
    elif kind == "oil":      # mode filter + posterize (painterly flat regions)
        out = _pil_map(x, lambda im: im.filter(ImageFilter.ModeFilter(5 if m else 7)).filter(ImageFilter.MedianFilter(5)))
        q = 32 if m else 40
        out = torch.floor(out / q) * q + q // 2
    else:
        raise ValueError(kind)
    return out.clamp(0, 255).round().to(torch.uint8)


def build(seed=0):
    idx = E.LONG[: 192 * 64]
    X = E.X[idx].clone(); y = E.y[idx]; seg = np.zeros(len(idx), int)
    g = torch.Generator().manual_seed(seed)
    s = 0
    for k, nb in enumerate(SEG_B):
        X[s:s + nb * 64] = corrupt(X[s:s + nb * 64], KINDS[k], g); seg[s:s + nb * 64] = k; s += nb * 64
    return X, y, seg


_CACHE = Path(__file__).resolve().parent / ("testlike_stream.pt" if not MILD else "testlike_stream_mild.pt")


def load():
    if _CACHE.exists():
        d = torch.load(_CACHE, weights_only=False)
        return d["X"], d["y"].numpy(), d["seg"].numpy()
    X, y, seg = build()
    torch.save({"X": X, "y": torch.from_numpy(y), "seg": torch.from_numpy(seg)}, _CACHE)
    return X, y, seg


def run(make, X, seed=0):
    """Stream X (uint8, already in stream order) through make(model) -> predictions."""
    from copy import deepcopy
    torch.manual_seed(seed); np.random.seed(seed)
    m = make(deepcopy(E.BASE))
    preds = []
    for s in range(0, len(X), 64):
        xb = X[s:s + 64].cuda().float()
        preds.append(m((xb / 255 - E.MEAN) / E.STD).cpu())
    return torch.cat(preds).numpy(), m


if __name__ == "__main__":
    X, y, seg = load()
    if sys.argv[1] == "--show":
        W = Image.new("RGB", (8 * 100, 12 * 100))
        s = 0
        for k, nb in enumerate(SEG_B):
            for j in range(8):
                W.paste(Image.fromarray(X[s + 200 + j * 7].permute(1, 2, 0).numpy()).resize((100, 100)), (j * 100, k * 100))
            s += nb * 64
        W.save(sys.argv[2]); sys.exit()
    from tta.arbn_teacher import AdaptiveRBNTeacher  # noqa: E402
    BASE_CFG = json.load(open(sys.argv[1]))
    jobs = [{"name": "base", "cfg": {}}] + json.loads(sys.argv[2])
    seeds = tuple(int(s) for s in sys.argv[3].split(",")) if len(sys.argv) > 3 else (0, 1)
    base = None
    for j in jobs:
        c = {**BASE_CFG, **j["cfg"]}; c["upscales"] = tuple(c.get("upscales", ()))
        cls = j.get("cls")
        acc, per = [], np.zeros((12, 2))
        for seed in seeds:
            pred, _ = run(lambda m: AdaptiveRBNTeacher(m, **c), X, seed)
            ok = pred == y; acc.append(ok.mean())
            for k in range(12):
                per[k] += [ok[seg == k].sum(), (seg == k).sum()]
        a = np.array(acc); base = a if base is None else base
        print(f"{j['name']:<32} {a.mean():.4f} paired {100 * (a - base).mean():+.2f} | "
              + " ".join(f"{KINDS[k][:5]} {per[k, 0] / per[k, 1]:.3f}" for k in range(12)), flush=True)
