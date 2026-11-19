"""Benchmark + diagnose every TTA method in starter/methods.py.

  ../.venv/bin/python analysis/analyze.py            (from the project root)

For each method:
  1. official run:  `run.py --split test` in its own process  -> wall-clock seconds, peak VRAM, test_pred.csv
  2. recorder runs: the official harness `run_stream` with a thin wrapper that stores softmax outputs and the
     per-batch compute time (val with seeds 0,1,2; test with seed 0).
Then computes per-domain / per-class / calibration / collapse / label-shift diagnostics and draws figures.

NOTE: val domain boundaries are recovered offline from image statistics (analysis only, never used by a method).
"""
import contextlib
import io
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "starter"
sys.path.insert(0, str(STARTER))
from harness import run_stream  # noqa: E402
from methods import METHODS  # noqa: E402
from run import load_model  # noqa: E402

DATA, CKPT, OUT, ANA = ROOT / "data", ROOT / "source_resnet50_160.pth", ROOT / "out", ROOT / "analysis"
PY = sys.executable
ORDER = ["source", "bn_adapt", "tent", "rbn", "rotta", "cotta", "rmt", "roid", "tribe"]
NAMES = {"source": "Source", "bn_adapt": "BN-Adapt", "tent": "TENT", "rbn": "Robust BN", "rotta": "RoTTA",
         "cotta": "CoTTA", "rmt": "RMT", "roid": "ROID", "tribe": "TRIBE"}
SEEDS = [0, 1, 2]
DOMAINS = ["scanline", "wave", "fisheye"]


# ------------------------------------------------------------------------------------------------ domains
def val_domains():
    f = ANA / "val_domains.csv"
    if f.exists():
        return pd.read_csv(f)
    ids = pd.read_csv(DATA / "val" / "stream.csv")["id"].tolist()
    dom = []
    for i in ids:
        x = np.asarray(Image.open(DATA / "val" / "images" / f"{i}.png").convert("L")).astype(float)
        corner = np.mean([x[:8, :8].mean(), x[:8, -8:].mean(), x[-8:, :8].mean(), x[-8:, -8:].mean()])
        line = np.abs(np.diff(x.mean(1), 2)).mean() + np.abs(np.diff(x.mean(0), 2)).mean()
        dom.append("fisheye" if corner < 3 else ("scanline" if line > 20 else "wave"))
    # domains are contiguous segments: absorb short runs (single mis-detections) into the previous run
    runs, s = [], 0
    for i in range(1, len(dom) + 1):
        if i == len(dom) or dom[i] != dom[s]:
            runs.append([s, i, dom[s]]); s = i
    merged = []
    for r in runs:
        if merged and (r[1] - r[0] < 16 or r[2] == merged[-1][2]):
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    dom = [d for a, b, d in merged for _ in range(a, b)]
    df = pd.DataFrame(dict(id=ids, domain=dom))
    df.to_csv(f, index=False)
    return df


# ------------------------------------------------------------------------------------------------ running
class Recorder:
    def __init__(self, method):
        self.m, self.probs, self.times = method, [], []

    def __call__(self, x):
        torch.cuda.synchronize(); t = time.perf_counter()
        p = self.m(x)
        torch.cuda.synchronize(); self.times.append(time.perf_counter() - t)
        self.probs.append(self.m.last_logits.float().softmax(1).cpu())
        return p


def official_test_run(name):
    out = OUT / name
    cmd = [PY, "run.py", "--method", name, "--split", "test", "--data", str(DATA), "--ckpt", str(CKPT), "--out", str(out)]
    subprocess.run(cmd, cwd=STARTER, check=True, capture_output=True, text=True)
    return json.load(open(out / "test_run_log.json"))


# ------------------------------------------------------------------------------------------------ metrics
def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def val_metrics(probs, y, dom, src_pred):
    pred, conf = probs.argmax(1), probs.max(1)
    ok = pred == y
    r = {"acc": ok.mean()}
    for d in DOMAINS:
        r[f"acc_{d}"] = ok[dom == d].mean()
    # adaptation speed: first 128 images after each shift vs rest of the domain
    starts = [0] + [i for i in range(1, len(dom)) if dom[i] != dom[i - 1]]
    early, late = [], []
    for s in starts:
        e = s + 128
        end = next((t for t in starts if t > s), len(dom))
        early.append(ok[s:e].mean()); late.append(ok[e:end].mean())
    r["acc_first128_after_shift"], r["acc_rest_of_domain"] = float(np.mean(early)), float(np.mean(late))
    r["mean_conf"], r["conf_correct"], r["conf_wrong"] = conf.mean(), conf[ok].mean(), conf[~ok].mean()
    r["ece"] = ece(conf, ok)
    # non-i.i.d. sensitivity: per-batch gain over source vs #true classes in the batch
    nb = len(y) // 64
    gain = [(ok[b * 64:(b + 1) * 64]).mean() - (src_pred[b * 64:(b + 1) * 64] == y[b * 64:(b + 1) * 64]).mean() for b in range(nb)]
    ncls = [len(np.unique(y[b * 64:(b + 1) * 64])) for b in range(nb)]
    r["corr_gain_vs_nclasses"] = float(np.corrcoef(gain, ncls)[0, 1])
    r["uniq_pred_per_batch"] = float(np.mean([len(np.unique(pred[b * 64:(b + 1) * 64])) for b in range(nb)]))
    r["uniq_true_per_batch"] = float(np.mean(ncls))
    # stream-level label distribution mismatch (total variation)
    hp, ht = np.bincount(pred, minlength=20) / len(y), np.bincount(y, minlength=20) / len(y)
    r["tv_pred_vs_true_hist"] = 0.5 * np.abs(hp - ht).sum()
    pc = np.array([ok[y == c].mean() for c in range(20)])
    r["per_class_acc"] = pc.round(3).tolist()
    return {k: (float(v) if not isinstance(v, list) else v) for k, v in r.items()}


def test_metrics(probs, ref_preds):
    pred, conf = probs.argmax(1), probs.max(1)
    nb = len(pred) // 64
    bconf = conf[: nb * 64].reshape(nb, 64).mean(1)
    hist = np.bincount(pred, minlength=20) / len(pred)
    h = -(hist[hist > 0] * np.log(hist[hist > 0])).sum() / np.log(20)
    r = {"conf_first20b": bconf[:20].mean(), "conf_last20b": bconf[-20:].mean(), "pred_hist_entropy": h,
         "max_class_share": hist.max(), "uniq_pred_per_batch": np.mean([len(np.unique(pred[b * 64:(b + 1) * 64])) for b in range(nb)])}
    for k, v in ref_preds.items():
        r[f"agree_{k}"] = (pred == v).mean()
    return {k: float(v) for k, v in r.items()}, bconf


# ------------------------------------------------------------------------------------------------ main
def main():
    dom_df = val_domains()
    val_ids = pd.read_csv(DATA / "val" / "stream.csv")["id"].tolist()
    y = pd.read_csv(DATA / "val" / "labels.csv").set_index("id").loc[val_ids, "label"].to_numpy()
    dom = dom_df.set_index("id").loc[val_ids, "domain"].to_numpy()

    results, curves, test_conf, test_preds = {}, {}, {}, {}
    src_pred = None
    for name in ORDER:
        print(f"== {name}", flush=True)
        off = official_test_run(name)
        res = {"test_seconds_official": off["seconds"], "test_peak_vram_mib": off["peak_vram_mib"],
               "test_torch_max_reserved_mib": off["torch_max_reserved_mib"]}
        seed_rows, times = [], []
        for s in SEEDS:
            probs, _, log, extra = recorded_run_timed(name, "val", s, times)
            if s == 0:
                if name == "source":
                    src_pred = probs.argmax(1)
                m0 = val_metrics(probs, y, dom, src_pred)
                curves[name] = (probs.argmax(1) == y).astype(float)
                res.update({f"val_seed0_{k}": v for k, v in extra.items()})
            seed_rows.append(val_metrics(probs, y, dom, src_pred))
        res["val"] = m0
        for k in ["acc"] + [f"acc_{d}" for d in DOMAINS]:
            vals = [r[k] for r in seed_rows]
            res[f"val_{k}_mean"], res[f"val_{k}_std"] = float(np.mean(vals)), float(np.std(vals))
        res["ms_per_batch_compute"] = 1000 * float(np.mean(times))
        tprobs, _, _, textra = recorded_run_timed(name, "test", 0, [])
        test_preds[name] = tprobs.argmax(1)
        refs = {k: test_preds[k] for k in ("source", "bn_adapt") if k in test_preds and k != name}
        tm, bconf = test_metrics(tprobs, refs)
        res["test"] = tm; res.update({f"test_{k}": v for k, v in textra.items()})
        test_conf[name] = bconf
        results[name] = res
        print(f"   val acc {res['val_acc_mean']:.4f}±{res['val_acc_std']:.4f}  test {off['seconds']}s  "
              f"VRAM {off['peak_vram_mib']} MiB", flush=True)

    json.dump(results, open(ANA / "results.json", "w"), indent=1)
    np.savez(ANA / "curves.npz", **{f"val_{k}": v for k, v in curves.items()}, **{f"testconf_{k}": v for k, v in test_conf.items()},
             y=y, dom=dom)
    print("saved", ANA / "results.json")


def recorded_run_timed(name, split, seed, times_sink):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rec = Recorder(METHODS[name](load_model(str(CKPT))))
    with contextlib.redirect_stdout(io.StringIO()):
        log = run_stream(rec, str(DATA), split, str(OUT / "_rec" / name), extra_log={"method": name, "seed": seed})
    probs = torch.cat(rec.probs).numpy()
    times_sink.extend(rec.times[1:])  # drop the first (cuDNN warm-up) batch
    extra = {k: getattr(rec.m, k) for k in ("skipped", "n_aug_batches") if hasattr(rec.m, k)}
    del rec; torch.cuda.empty_cache()
    return probs, None, log, extra


if __name__ == "__main__":
    main()
