"""Figures for analysis/results.json + curves.npz (run analyze.py first)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANA = Path(__file__).resolve().parent
R = json.load(open(ANA / "results.json"))
C = np.load(ANA / "curves.npz", allow_pickle=True)
ORDER = ["source", "bn_adapt", "tent", "rbn", "rotta", "cotta", "rmt", "roid", "tribe"]
NAMES = {"source": "Source", "bn_adapt": "BN-Adapt", "tent": "TENT", "rbn": "Robust BN", "rotta": "RoTTA",
         "cotta": "CoTTA", "rmt": "RMT", "roid": "ROID", "tribe": "TRIBE"}
BLUE, GRAY, INK, INK2, GRID, SURF = "#2a78d6", "#9a9893", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
         "#1c5cab", "#184f95", "#104281", "#0d366b"]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})

dom = C["dom"]
bounds = [i for i in range(1, len(dom)) if dom[i] != dom[i - 1]]
segs = [0] + bounds + [len(dom)]


def batch_mean(v, bs=64):
    n = len(v) // bs
    return v[: n * bs].reshape(n, bs).mean(1)


# 1. accuracy along the val stream (small multiples, source as gray reference)
fig, axes = plt.subplots(3, 3, figsize=(11, 7.5), sharex=True, sharey=True)
src = batch_mean(C["val_source"])
xb = np.arange(len(src)) * 64 + 32
for ax, m in zip(axes.flat, ORDER):
    ax.grid(axis="y", color=GRID, lw=0.8)
    for b in bounds:
        ax.axvline(b, color=INK2, lw=1, ls="--")
    if m != "source":
        ax.plot(xb, src, color=GRAY, lw=1.5, label="Source")
    ax.plot(xb, batch_mean(C[f"val_{m}"]), color=BLUE, lw=2, marker="o", ms=3, label=NAMES[m])
    ax.set_title(f"{NAMES[m]}  (val {R[m]['val_acc_mean']:.3f})", loc="left", fontsize=10)
    ax.set_ylim(0, 1)
    for a, b in zip(segs[:-1], segs[1:]):
        ax.text((a + b) / 2, 0.04, dom[a], ha="center", color=INK2, fontsize=8)
h, l = axes[0, 1].get_legend_handles_labels()
fig.legend(h, ["Source (no adaptation)", "method"], loc="upper right", ncol=2, frameon=False, fontsize=9)
for ax in axes[-1]:
    ax.set_xlabel("image index in val stream")
for ax in axes[:, 0]:
    ax.set_ylabel("batch accuracy")
fig.suptitle("Per-batch accuracy along the val stream (dashed = domain shift; gray = no adaptation)", x=0.01,
             ha="left", color=INK)
fig.tight_layout()
fig.savefig(ANA / "fig1_val_stream_accuracy.png", dpi=130)

# 2. method x domain heatmap
cols = ["acc", "acc_scanline", "acc_wave", "acc_fisheye"]
labels = ["overall", "scanline", "wave", "fisheye"]
M = np.array([[R[m][f"val_{c}_mean"] for c in cols] for m in ORDER])
fig, ax = plt.subplots(figsize=(6, 5))
lo, hi = 0.3, 0.9
idx = np.clip(((M - lo) / (hi - lo) * (len(BLUES) - 1)).round().astype(int), 0, len(BLUES) - 1)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.add_patch(plt.Rectangle((j, i), 0.96, 0.92, color=BLUES[idx[i, j]]))
        ax.text(j + 0.48, i + 0.46, f"{M[i, j]:.3f}", ha="center", va="center",
                color="white" if idx[i, j] >= 7 else INK, fontsize=9)
ax.set_xlim(0, 4); ax.set_ylim(len(ORDER), 0)
ax.set_xticks(np.arange(4) + 0.48, labels); ax.set_yticks(np.arange(len(ORDER)) + 0.46, [NAMES[m] for m in ORDER])
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
ax.set_title("Val accuracy by corruption domain (mean of 3 seeds)", loc="left")
fig.tight_layout()
fig.savefig(ANA / "fig2_domain_heatmap.png", dpi=130)

# 3. mean confidence along the test stream
fig, axes = plt.subplots(3, 3, figsize=(11, 7.5), sharex=True, sharey=True)
srcc = C["testconf_source"]
for ax, m in zip(axes.flat, ORDER):
    ax.grid(axis="y", color=GRID, lw=0.8)
    if m != "source":
        ax.plot(srcc, color=GRAY, lw=1.2, label="Source")
    ax.plot(C[f"testconf_{m}"], color=BLUE, lw=1.6, label=NAMES[m])
    note = "  (logits × prior: not a probability)" if m == "roid" else ""
    ax.set_title(NAMES[m] + note, loc="left", fontsize=10)
    ax.set_ylim(0, 1)
h, l = axes[0, 1].get_legend_handles_labels()
fig.legend(h, ["Source (no adaptation)", "method"], loc="upper right", ncol=2, frameon=False, fontsize=9)
for ax in axes[-1]:
    ax.set_xlabel("test batch index (192 batches)")
for ax in axes[:, 0]:
    ax.set_ylabel("mean max-softmax")
fig.suptitle("Unlabeled test stream: mean prediction confidence per batch (jumps = domain shifts)", x=0.01,
             ha="left", color=INK)
fig.tight_layout()
fig.savefig(ANA / "fig3_test_confidence.png", dpi=130)

# 4. cost vs limits: two separate charts, no dual axis
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
names = [NAMES[m] for m in ORDER][::-1]
secs = [R[m]["test_seconds_official"] for m in ORDER][::-1]
vram = [R[m]["test_peak_vram_mib"] / 1024 for m in ORDER][::-1]
for ax, vals, lim, unit, title in [(a1, secs, 90, "s", "Test-stream wall-clock on RTX 4090 (limit 90 s on a 3090)"),
                                   (a2, vram, 10, "GB", "Peak VRAM, nvidia-smi (limit 10 GB)")]:
    ax.barh(names, vals, color=[BLUE if v <= lim else "#e34948" for v in vals], height=0.6)
    ax.axvline(lim, color=INK, lw=1.2, ls="--")
    ax.text(lim, len(names) - 0.4, f" limit {lim} {unit}", color=INK, fontsize=8, va="bottom")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.1f}", va="center", color=INK2, fontsize=8)
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.tick_params(axis="y", length=0)
fig.tight_layout()
fig.savefig(ANA / "fig4_cost.png", dpi=130)
print("figures written to", ANA)
