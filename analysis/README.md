# Baseline benchmark of 8 TTA methods (original settings, no tuning)

Reproduce (from project root, ~10 min on an RTX 4090):

```bash
.venv/bin/python analysis/analyze.py   # runs everything -> results.json, curves.npz, out/<method>/test_pred.csv
.venv/bin/python analysis/plots.py     # -> fig1..fig4 PNGs
```

Single method, official harness:
`cd starter && ../.venv/bin/python run.py --method rbn --split val --data ../data --ckpt ../source_resnet50_160.pth --out ../out/rbn`

| Method | Val acc (3 seeds) | scanline | wave | fisheye | first 128 imgs after shift | Test time (4090) | Peak VRAM |
|---|---|---|---|---|---|---|---|
| Source | 0.546 | 0.386 | 0.494 | 0.812 | 0.589 | 4.3 s | 1.0 GB |
| BN-Adapt | 0.722 | 0.802 | 0.712 | 0.631 | 0.698 | 4.5 s | 1.0 GB |
| TENT | 0.723 | 0.802 | 0.712 | 0.634 | 0.701 | 8.3 s | 3.6 GB |
| **Robust BN** | **0.762** | 0.840 | 0.668 | 0.775 | 0.633 | 6.0 s | 1.2 GB |
| RoTTA | 0.650 | 0.812 | 0.331 | 0.820 | 0.513 | 25.6 s | 8.4 GB |
| CoTTA | 0.723 | 0.803 | 0.712 | 0.634 | 0.698 | 17.0 s | 4.0 GB |
| RMT* | 0.750 ± 0.005 | 0.806 | 0.724 | 0.709 | 0.729 | 23.6 s | 6.5 GB |
| ROID | 0.721 | 0.802 | 0.709 | 0.632 | 0.701 | 15.4 s | 7.0 GB |
| TRIBE | 0.734 | 0.844 | 0.528 | 0.836 | 0.508 | 30.8 s | **16.4 GB ✗** |

Val domains (recovered offline from image statistics, analysis only): scanline 0–776, wave 777–1499,
fisheye 1500–2111. *RMT is the source-free variant (no warm-up/replay; fc weights as prototypes).
TRIBE skipped every update (entropy filter H0·ln20 = 0.15 nats; the lowest teacher entropies were about 0.64 nats).

## Adaptive Robust BN (`--method arbn`)

alpha_t = clip(GAIN · s_t, ALPHA_MIN, ALPHA_MAX), where s_t = symmetric KL between the batch statistics and the
EMA statistics on stem + layer1 (measured before the update). Sweep: `.venv/bin/python analysis/sweep_arbn.py`
-> `sweep_arbn.csv` (90 adaptive configs + 8 fixed alphas; official val order + all 6 domain orders).

| Config | val (official) | 6-order mean | 6-order worst | first 128 after shift | test time |
|---|---|---|---|---|---|
| Robust BN α=0.05 (RoTTA default) | 0.762 | 0.767 | 0.745 | 0.516 | 6.0 s |
| Robust BN α=0.2 (best fixed) | 0.801 | 0.792 | 0.779 | 0.723 | – |
| **arbn shallow, min 0.05, max 0.5, gain 1.0 (default)** | **0.825** | **0.819** | **0.812** | **0.742** | 8.1 s |
| arbn shallow, min 0.02, max 1.0, gain 2.0 (best mean) | 0.827 | 0.826 | 0.807 | 0.703 | – |

## arbn × TENT × LAME (2×2), `analysis/lame_analysis.py` -> `lame_analysis.log`

| | no LAME | LAME (k=5) |
|---|---|---|
| arbn | 0.825 / 0.819 / 0.812 / 0.813 | 0.857 / 0.852 / 0.844 / 0.847 |
| arbn + TENT (SGD lr 0.1) | 0.852 / 0.838 / 0.820 / 0.878 | **0.874 / 0.863 / 0.849 / 0.892** |

(official val / 6-order mean / 6-order worst / long stream). Test time on an idle 4090: arbn_lame 8 s / 1.2 GB,
arbn_tent 19 s / 6.5 GB. LAME changes: neighbour agreement predicts precision (5/5 ≈ 10:1 fixes:breaks,
≤2/5 ≈ 3:1), base margin does not; no (margin × agreement) cell is net harmful, and leave-one-domain-out
gating picks "always LAME" in every fold. Phase-B safeguards (entropy filter, pull-back to source, deep-only)
did not help at lr ≤ 0.05; predicting after the update gives about +0.3.

## Follow-ups (2026-11-13): T3A, mean teacher, unfreezing

Final candidate: `--method arbn_teacher` = arbn + TENT (SGD lr 0.1, BN γ/β only) + mean teacher (EMA 0.999,
symmetric CE on a strongly augmented view) + T3A (50 supports/class) + LAME (k=5), prediction from the student.

| Method (seeds) | val | 6-order mean | 6-order worst | long |
|---|---|---|---|---|
| arbn_tent + LAME (det.) | 0.874 | 0.863 | 0.849 | 0.892 |
| arbn_tent + T3A + LAME (det.) | 0.887 | 0.876 | 0.860 | 0.900 |
| arbn_teacher + LAME (3) | 0.887 ± .003 | 0.874 ± .001 | 0.862 ± .002 | 0.912 ± .002 |
| **arbn_teacher + T3A + LAME (3)** | **0.890 ± .003** | **0.881 ± .002** | **0.865 ± .004** | **0.914 ± .002** |

Test stream on an idle 4090: 33–34 s, 6.8 GB (two separate backward passes keep VRAM down from 9.0 GB).
Teacher with EMA 0.999 stays ≈ arbn (its own accuracy 0.825): it acts as an anchor + augmentation-consistency
target, not as a better predictor. Unfreezing layer4.2 / layer4 (conv lr 1e-4…1e-2): no gain, large lr hurts
(`sweep_unfreeze.log`). T3A stale handling (keep / window / reset) makes no difference (`t3a_sim.log`).

## Error analysis, flip view, feedback (2026-11-13)

`error_analysis.py`: final arbn_teacher top-1 0.888 / top-3 0.963 (official order). Errors per val run ≈ 236:
truth 2nd + a signal supports it ≈ 50, truth in top-3 but every signal agrees on the same wrong label ≈ 60,
truth outside top-3 ≈ 80–95 (mostly fisheye; laptop, timber_wolf). "T3A and LAME both support a change": ≈ 30
images/run, 20 fixes / 3 breaks, agreed label correct only 67%.

`flip_paired.py` (same seed + same order, 21 pairs): flip-view averaging (original + horizontal flip, eval mode,
BN statistics verified untouched) improves the final output in 21/21 pairs, +1.6 points on average (min +0.2).
Fixes ≈ 63 / breaks ≈ 30 errors per run. Of the base "truth outside top-3" errors, the flip path brings the truth
into the top-3 for 31.5% (fisheye: 24.5%) and to top-1 for 8.7% (fisheye: 5.3%).

`sweep_feedback.py` (3 seeds, short streams only):

| Config | val | 6-order mean | 6-order worst | first 128 after shift |
|---|---|---|---|---|
| base (arbn_teacher) | 0.890 | 0.881 | 0.865 | 0.789 |
| **+ flip (new default)** | **0.909** | **0.896** | **0.885** | **0.823** |
| + feedback "both" w 0.25–1.0 | 0.887–0.890 | 0.879–0.881 | 0.862–0.865 | 0.788–0.790 |
| + feedback "all" w 0.25–1.0 | 0.881–0.884 | 0.872–0.879 | 0.857–0.864 | 0.793–0.809 |
| + flip + feedback "both" 0.5 | 0.909 | 0.896 | 0.887 | 0.821 |

Test stream (idle 4090): 35.8–36.0 s, 6.9 GB. Harness val with flip: 0.914. Predictions: `out/arbn_teacher_flip/`.

## Center-crop prediction views (2026-11-13): `crop_paired.py`, `views_sim.py`

Extra prediction views (eval mode, after the statistics update, before the SGD step; no effect on adaptation).
On the "truth outside top-3" fisheye errors of the original+flip version, a 70% center crop alone puts the truth
in the top-3 for 68.6% (top-1 46.6%) vs 11.3% (0.1%) for the flip. Crops alone hurt scanline (C70 single-view
0.665 vs 0.871), so the rule matters. Dark-corner gate (four 12x12 corners, mean < 30 on 0-255): fisheye 100%,
scanline / wave < 1%. Entropy-weighted view averaging was worse than plain averaging.

| Views (3 seeds × 7 streams, paired vs OF) | val | 6-order mean | 6-order worst | scan | wave | fish |
|---|---|---|---|---|---|---|
| O + F (previous default) | 0.909 | 0.896 | 0.885 | 0.918 | 0.905 | 0.865 |
| O + F + C70 + CF70 | 0.933 | 0.920 | 0.911 | 0.919 | 0.911 | 0.940 |
| O + F + C60 + CF60 only for dark-corner images | 0.929 | 0.920 | 0.912 | 0.919 | 0.906 | 0.942 |
| **O + F + C80 + CF80, + C60 + CF60 for dark-corner images (new default)** | **0.938** | **0.928** | **0.918** | **0.928** | **0.914** | **0.949** |

New default: paired +3.1 points over O+F (min +2.4, 21/21). Harness val 0.942. Test stream (idle 4090):
44.2–44.3 s, 6.9 GB. Predictions: `out/arbn_teacher_views/`.

## Kaggle public calibration (2026-11-13)

| Version (`out/...`) | val (harness) | public | gap | step on val | step on public |
|---|---|---|---|---|---|
| arbn | 0.825 | 0.80729 | −1.8 | | |
| arbn_tent (TENT lr 0.1 + LAME) | 0.874 | 0.85774 | −1.6 | +4.9 | +5.1 |
| arbn_teacher (no extra views) | 0.893 | 0.88509 | −0.8 | +1.9 | +2.7 |
| **arbn_teacher_flip (current default)** | **0.914** | **0.91048** | −0.4 | +2.1 | +2.5 |
| arbn_teacher_views (+C80 all, +C60 dark corners) | 0.942 | 0.90511 | −3.7 | +2.8 | −0.5 |

The test stream has no fisheye-like domain: only 3.5% of test images have dark corners and no batch is >50% dark.
Crop views were a val-specific gain and hurt on public, so the default is back to original + flip
(`CROPS = ()`, `DARK_CROPS = ()`; reproduces the 0.91048 file exactly, 36 s on the 4090). TENT, the teacher + T3A
and the flip view transfer to public at least as well as val predicted.

## Image bank / full RoTTA memory in the consistency step (2026-11-14): `sweep_bank.py`, `analyze_bank.py`

A none (default) / B recent replay (48 current + 16 from a FIFO of 256, stored teacher targets) / C class-balanced
replay / D RoTTA CSTU memory (64, targets recomputed by the current teacher, time-aware weights) / D + reset on shift.
Short val streams (3 seeds × 7): all within noise — D best with paired +0.22 (14/21), scanline/wave +0.5, few-class
batches (≤ 8 classes) +0.65, but fisheye −0.4 and after-shift −0.4; reset on shift changes nothing. Long val stream
(198 batches): no collapse for any variant (0.928–0.933).

**On the real test stream every bank variant collapses** (second half: 3–4.4 classes per batch, largest class
25–40%, agreement with the 0.91048 file 38–45%). Likely mechanism: replayed images from earlier domains are normalized
with the current domain's arbn statistics; the consistency loss on these off-distribution outputs, with SGD lr 0.1,
drags γ/β away. The test stream has far larger shifts than val (shallow KL up to 7 vs 1.2), so val cannot show it.
Collapsed outputs were renamed `out/_COLLAPSED_*`. Default stays `BANK = None`.

`health_check.py out/<v>/test_pred.csv`: label-free collapse check on the test stream — run it before every submission.

## Learning rate × image bank on Kaggle public (2026-11-14)

| Version (`out/...`) | val (harness) | public |
|---|---|---|
| arbn_teacher_flip (lr 0.1, no bank) — best | 0.914 | **0.91048** |
| arbn_teacher_lr001_nobank (lr 0.01) | 0.886 | 0.89143 |
| arbn_teacher_rotta_lr001 (lr 0.01, RoTTA bank) | 0.880 | 0.88736 |
| arbn_teacher_rotta_oldw01 (lr 0.1, RoTTA bank, old images ×0.1) | 0.912 | 0.89811 |

The bank lowers public in both settings (−0.4 at lr 0.01, −1.2 at lr 0.1): dropped. lr 0.1 beats lr 0.01 on public
(+1.9), consistent with val.

## History experiments (2026-11-14): `history_exp.py`, `analyze_history.py` -> `analyze_history.log`

Aligned val streams (domains cut to 768 / 704 / 576 so boundaries fall on batch boundaries; identical batch
companions in all 6 orders), 3 seeds.
- Same image, different history: history is net neutral-to-helpful (harmed 1.0–2.3%, helped 0.9–3.9%, always wrong
  3.9–9.5% per domain/predecessor). Batches 0–1 of a domain are weak even as the first segment (0.828 / 0.830 vs
  0.955 at batch 2), so most of the "after-shift" gap is batch content / no adaptation yet, not harmful history.
- Momentum vs fresh gradient: cosine 0.10 at the shift batch, 0.1–0.2 elsewhere (never opposed).
- B = clear SGD momentum at a detected onset (alpha ≥ 0.3): standard streams paired +0.20 (12/21, min −0.14),
  after-shift +0.45, scanline/wave +0.35. Test: 36.5 s, 12 resets, healthy (agreement 0.952 with the best file):
  `out/arbn_teacher_momreset/`.
- Historical neighbours: on images the final gets wrong, a history kNN vote is right only 8.8% (≤ 16 fixes per
  run) while it disagrees with a correct final on 141 per run → no useful extra information.
- Steady part (batch ≥ 2), base → T3A → LAME: RRR 1460, WWW 125 (7.5%), WWR 40, WRR 19, RRW 6, RWR 7 per run;
  any stage right 0.925 vs final 0.917 → output fusion can add at most ~0.8.

Public: arbn_teacher_momreset (momentum reset at shift) 0.90543 (val +0.2, public −0.5 vs 0.91048) → not adopted.

## Augmentation decomposition (2026-11-14): `aug_diag.py`, `sweep_auggroups.py`, `sweep_aug.py`

Diagnostic views on the default (same model state as the answer, eval mode, RNG restored; aligned streams, 3 seeds):

| view | student acc | right→wrong | wrong→right | hard WWW rescued |
|---|---|---|---|---|
| original | 0.857 | – | – | 6.6% |
| color only | 0.756 | 14.7% | 17.2% | 13.1% |
| geometric only | 0.767 | 15.3% (scanline 26.7%) | 28.3% | 20.7% |
| blur + noise only | 0.853 | 1.7% | 7.4% | 8.7% |
| full strong aug | 0.638 | 29.5% (scanline 43.1%) | 23.2% | 17.5% |

Training variants (standard streams, 3 seeds, paired vs A): 4 groups × 16 augmentation draws +0.20 (12/21);
soft −0.02; **no_geom +0.28 (12/21), scanline 0.918→0.929, scanline+wave 0.912→0.919, fisheye −0.7**; no_color −0.25;
weak (blur+noise+flip) −0.25 overall but scanline+wave 0.919 (wave 0.917), fisheye −2.5.
Healthy test files: `out/arbn_teacher_aug_no_geom/` (34.6 s), `out/arbn_teacher_aug_weak/` (35.7 s).

Public: arbn_teacher_aug_no_geom **0.91243 (new best, now the default `AUG = "no_geom"`)**; arbn_teacher_aug_weak 0.90250.

## Local pooling vs whole-image upscaling (2026-11-14)

`local_feat_diag.py`: re-pooling the 5×5 layer4 map (center 3×3, center-weighted 2:1 / 4:1, source fc) rescues only
1–2% of the hard errors (≈ 1–3 of 119 per run); crop re-run rescued ~47% of fisheye hard errors → the crop effect is
object scale, not "where to look".

`upscale_paired.py` (default no_geom; 3 seeds × 7, paired vs O+F):

| views | val | 6-order mean / worst | scan | wave | fish | scan+wave | paired |
|---|---|---|---|---|---|---|---|
| O + F (default) | 0.910 | 0.900 / 0.887 | 0.929 | 0.908 | 0.857 | 0.919 | – |
| + U192 | 0.923 | 0.913 / 0.900 | 0.937 | 0.914 | 0.885 | 0.926 | +1.3 (21/21) |
| + U224 | 0.926 | 0.915 / 0.904 | 0.936 | 0.914 | 0.893 | 0.926 | +1.5 (21/21) |
| + U192 + UF192 | 0.928 | 0.915 / 0.901 | 0.935 | 0.915 | 0.897 | 0.925 | +1.6 (21/21) |
| + U224 + UF224 | 0.934 | 0.921 / 0.907 | 0.934 | 0.913 | 0.920 | 0.924 | +2.2 (21/21) |

Single U224 view: rescues 37.5% of hard errors to top-1 but scanline alone 0.600. Healthy test files:
`out/arbn_teacher_up192/` (40.5 s, 7.4 GB), `out/arbn_teacher_up224/` (42.4 s, 8.4 GB). Option `upscales=(192,)`.

## Autonomous exploration (2026-11-14): see `analysis/AUTO_SUMMARY.md`

Final `--method arbn_best` (`starter/best_config.json`): soft consistency aug + views O + F + U224 + UF224 +
fused Robust BN + fp16 views. Val (3 seeds × 7): official 0.936, all-mean 0.925, 6-order worst 0.907, scan+wave 0.929,
paired +2.4 (21/21) vs the public-best default; test 32.7 s / 4.5 GB on the 4090, healthy. `out/arbn_best/`.
Stage 2 of the search accepted only aug=soft; lr / w_ent / w_cons / mt / T3A K / LAME K / alpha_max / gain were all
within noise or worse; removing LAME (−3.4) or T3A (−0.8) hurts clearly.

Public: arbn_best (soft aug, O+F+U224+UF224, fused BN, fp16 views) **0.91682 — new best** (+0.44 over 0.91243).
Public: arbn_teacher_up192 (no_geom, O+F+U192) 0.91536 → the upscale view itself transfers (+0.29 over 0.91243);
soft aug + U224/UF224 add +0.15 more (arbn_best 0.91682).

## Where can more accuracy come from? (2026-11-14, later): ceilings + temporal LAME

Stream structure (val truth; test predictions show the same statistics): consecutive images come from a few classes
(4.5 distinct classes within ±4 positions vs 7.3 if shuffled; P(same label as previous) 0.25 vs 0.06). Classes are
balanced per whole domain only — half a domain is already 26–32% skewed, so a class-balance prior has little to use.

Ceilings measured on val (arbn_best = 0.936 official, 0.925 over 3 seeds × 7 streams):

| Experiment | Script | Result |
|---|---|---|
| True labels drive the γ/β updates (supervised CE instead of entropy) | `oracle_exp.py` | final +0.8 only (0.928 → 0.936, seed 0, 7 streams); layer4 unfrozen: same |
| Adaptation lag after a shift (each domain streamed with circular shifts) | `roll_exp.py` | 3.6 images per domain (0.6%) after another domain; 11 images from the source model |
| Whole pipeline at base 176 / 192 / 208 / 224 / 256 (stats adapted there) | `res_exp.py` | scan+wave stays 0.92–0.937; only fisheye moves. 192 with O+F ≈ 160 with 4 views |
| Temporal prior with the TRUE labels of the ±w stream neighbours + LAME | `temporal_oracle.py` | 0.941 all-mean (vs 0.9225 simulated base) |
| Class-balance logit adjustment (running marginal) | `temporal_sim3.py` | +0.3, not kept |
| T3A with refined (final) labels, T3A + fc logits | `t3a_sim2.py` | −0.1 … −0.9 |
| LAME graph: centering, mutual kNN, soft weights, temperature, graph weight, previous-batch context | `temporal_sim.py`, `temporal_sim4.py` | all worse; k=5 / weight 1 / temp 1 is a sharp optimum |

Adopted candidates (LAME graph only; adaptation untouched, defaults unchanged so `arbn_best` reproduces exactly):
`lame_time` / `lame_tau` = weight of an extra affinity between images at most tau stream positions apart in the batch;
`lame_pow` = square root of the features before the kNN search.

| Version | paired, 3 seeds × 7 (`lame_paired.py`) | scan+wave | fisheye | harness val, seeds 0/1/2 | test (4090) |
|---|---|---|---|---|---|
| arbn_best | – | 0.928 | 0.917 | 0.9366 / 0.9347 / 0.9370 = 0.9361 | 33.9 s, 4.5 GB |
| time 0.1·box(8) (`config_temporal_nopow.json`) | +0.44 (20/21) | 0.934 | 0.919 | 0.9432 / 0.9384 / 0.9441 = **0.9419** | 33.9 s, healthy, 98.0% agreement with arbn_best |
| pow + time 0.1·box(8) (`config_temporal.json`) | +0.60 (21/21) | 0.938 | 0.914 | 0.9394 / 0.9399 / 0.9384 = 0.9392 | 34.6 s, healthy, 95.4% agreement |

Reproduce: `cd starter && ARBN_CONFIG=$PWD/config_temporal_nopow.json ../.venv/bin/python run.py --method arbn_best --split test --data ../data --ckpt ../source_resnet50_160.pth --out ../out/arbn_temporal_nopow`
Files awaiting public: `out/arbn_temporal_nopow/test_pred.csv`, `out/arbn_temporal/test_pred.csv`.

## Learn first, then predict (2026-11-14, evening): `cfg_paired.py`, `predict_after*.log`

Public: arbn_temporal (pow + time 0.1·box(8)) **0.92692 — new best** (+1.0 over arbn_best 0.91682).

`predict_after=True`: the SGD step on the batch comes first, then every prediction view (original included) goes
through the updated student (eval mode, statistics already updated by this batch). `extra_steps=n`: n more SGD steps
on the same batch before answering (eval-mode forwards, new augmentation draw); `extra_loss` "both" | "ent" | "cons";
`extra_stride=2`: each extra step uses every 2nd image; `extra_revert`: undo the extra steps after answering.
Paired vs arbn_temporal, 3 seeds × 7 streams:

| Config | paired | official (evalkit) | 6-order worst | scan+wave | harness val, 3 seeds | test time (4090) |
|---|---|---|---|---|---|---|
| arbn_temporal (predict, then learn) | – | 0.9421 | 0.915 | 0.938 | 0.9392 | 34–36 s |
| learn first (`config_learnfirst.json`) | +0.29 (16/21) | 0.9438 | 0.921 | 0.942 | 0.9416 | 34–37 s |
| + 1 extra step, both losses | +0.71 (21/21) | 0.9441 | 0.928 | 0.946 | – | – |
| + 2 extra steps, both losses (`_x2`) | +0.95 (19/21) | 0.9506 | 0.928 | 0.951 | – | 66 s (too slow) |
| + 3 / 4 extra steps, both losses | +0.75 / +0.74 | 0.946 / 0.945 | 0.925 / 0.927 | 0.950 / 0.951 | – | – |
| + 1 extra, consistency only (`_x1cons`) | +0.62 (19/21) | 0.9460 | 0.926 | 0.945 | 0.9462 | 47–49 s |
| + 2 extra, consistency only (`_x2cons`) | +0.79 (18/21) | 0.9471 | 0.931 | 0.951 | 0.9531 (seed 0) | 50–53 s |
| + 2 extra, consistency, half batch (`_h2`) | +0.78 (17/21) | 0.9462 | 0.930 | 0.949 | **0.9476** | 45–50 s |
| + 3 extra, consistency, half batch (`_h3`) | +0.88 (19/21) | 0.9473 | 0.929 | 0.952 | 0.9448 | 49–56 s |
| + 2 extra, entropy only | −0.21 | 0.931 | 0.915 | 0.938 | – | – |
| + 2 / 4 extra, both losses, reverted after answering | +0.11 / −0.41 | 0.939 / 0.938 | 0.922 / 0.918 | 0.940 / 0.936 | – | – |
| 1 extra step but predict before (control) | −0.01 | 0.941 | 0.918 | 0.937 | – | – |
| learn first, main entropy loss removed (w_ent 0) | −0.42 | 0.936 | 0.915 | 0.936 | – | – |
| + 2 / 3 extra consistency, w_ent 0 | +0.73 / +0.47 | 0.945 / 0.943 | 0.931 / 0.927 | 0.950 / 0.949 | – | – |

Gain needs both more learning on the batch and answering after it (control and revert both lose it); consistency is
the useful extra loss; saturates at 2 extra steps. One full-batch extra consistency step ≈ 43 ms/batch; fp16 and
channels_last are not faster; test-stream timing varies by ±3–5 s between identical runs. All test files healthy
(agreement with arbn_temporal 0.92–0.97): `out/arbn_learnfirst{,_x1cons,_h2,_h3,_x2cons,_x2}/`.

## After public 0.94433 (2026-11-14, night): speed, more steps, harsh proxy

Public: arbn_learnfirst_h2 (learn first + 2 extra consistency steps on half the batch) **0.94433 — new best**
(+1.74 over arbn_temporal; val had predicted +0.78).

Speed: a forward+backward is dominated by per-layer overhead, not GPU load (64 images 38 ms, 32 images 32 ms).
`fast_bn=True` normalizes with one `F.batch_norm` call in eval mode: whole call 284 → 223 ms, test 45–50 → 41–42 s.
Same math up to rounding (paired −0.08, 9/21), but 3.4% of the test predictions differ from the 0.94433 file
→ public differences below about ±0.3 between versions are not meaningful. fp16 / channels_last: not faster.

Paired vs h2 (3 seeds × 7 streams, `predict_after4.log`): 2 full steps +0.12; 4 half / 3 full steps −0.02 / −0.07;
extra-step target = provisional T3A+LAME answer −7.9 (collapse, worst order 0.757); target = mix +0.21 (fisheye only).

`harsh_val.py`: val stream with a second synthetic corruption changing every 352 images (labelled proxy for large
shifts; `PROXY=mild` for a weaker version). Accuracy, paired vs arbn_temporal, 3 seeds × 2 streams:

| Config | harsh proxy | mild proxy (false colour excluded: 0.17 for every method) |
|---|---|---|
| arbn_temporal | 0.492 | 0.860 |
| learn first, no extra step | 0.490 | – |
| h2 (2 half steps) | 0.560 (+6.8) | 0.872 |
| 2 / 3 / 4 full steps | 0.561 / 0.567 / 0.580 | 0.873 / – / 0.870 |
| 4 half steps | 0.585 | 0.871 |
| 2 full steps, target mix | 0.528 (min −6.2) | – |
| h2 + 4 more steps when alpha ≥ 0.3 | 0.585 (+9.3) | 0.876 |
| h2 + 8 more steps when alpha ≥ 0.3 | **0.592 (+10.0)** | **0.878** |

`shift_steps=n`, `shift_thr=0.3`: n more (full-batch) extra steps on batches whose arbn momentum alpha_t ≥ 0.3, i.e.
while a shift is being absorbed (decided from the current batch's statistics, no batch index). Standard val: neutral
(paired −0.02 / +0.02 for 4 / 8). On the test stream it fires on 19–20 of 192 batches.

| Version | harness val, seeds 0/1/2 | test (4090) | health |
|---|---|---|---|
| `config_h2_fast.json` → `out/arbn_h2_fast/` | 0.9493 / 0.9498 / 0.9470 = 0.9487 | 41–42 s, 4.4 GB | ok, 96.6% agreement with h2 |
| `config_x2_fast.json` → `out/arbn_x2_fast/` (2 full steps) | 0.9493 / 0.9446 / 0.9489 = 0.9476 | 48–49 s | ok |
| `config_h2_shift4.json` → `out/arbn_h2_shift4/` | 0.9451 / 0.9375 / 0.9455 = 0.9427 | 41–42 s | ok, 94.7% |
| `config_h2_shift8.json` → `out/arbn_h2_shift8/` | 0.9522 / 0.9413 / 0.9512 = 0.9482 | 43–47 s | ok, 94.3% |

## Public results for the fast-BN versions and run-to-run spread (2026-11-15)

Public: arbn_h2_fast 0.93782, arbn_h2_shift8 0.93766 (arbn_learnfirst_h2, same method on the original BN path: 0.94433).

Is fast BN worse? Checks: same model state and inputs → logits differ ≤ 0.013 (fp32) and fp16 views are no further
from fp32 than the original path; all finite. Labelled A/B vs the original path: val −0.08 (9/21), mild proxy −0.21
(2/6), harsh proxy +0.68 (4/6), long val stream (4 seeds) 0.9474 vs 0.9484. No systematic loss.

Run-to-run spread (`seed_noise.log`; long val stream = 12,672 images, 18 shifts, 4 seeds): two seeds of the same
config disagree on 3.1–3.3% of the images (h2 vs h2_fast on test: 3.4%), and accuracy on a 6,336-image half ranges
over 0.77–0.82 points across the 4 seeds. → A 0.65 gap between two single public scores is inside the normal spread;
single public submissions cannot rank versions that differ by less than about 0.8. The expected public level of the
h2 family is about 0.94; 0.94433 was a draw on the high side. Shift-triggered steps: no public evidence of a gain.

## Averaging independent runs (2026-11-15): `ensemble_exp.py`

Final probabilities of several seeds of h2_fast averaged per image (each run is its own online pass). Labelled streams,
6 seeds: val official 0.9470 → 0.9482 / 0.9493 / 0.9508 (2 / 4 / 6 runs); long val 0.9471 → 0.9497 / 0.9512 / 0.9511;
mild proxy 0.7476 → 0.7563 / 0.7628 / 0.7637. On the test stream two seeds disagree on 5.3% of images (long val 3.3%).
Probe files, NOT valid final submissions (several passes over the test stream, not reproducible by one run):
`out/probe_avg2/`, `out/probe_avg4/` (h2_fast seeds 0–1 / 0–3). A rule-conforming version would have to run the
models side by side in one pass, i.e. about 2× / 4× the time of h2_fast (41 s on the 4090).
Three-view variant (no UF224): harness val 0.9402 (−0.85), saves only 1–2 s → not used.

## Tuning and ablation around arbn_learnfirst_h2 (2026-11-15): `ablation.py`, `tune_h2.md`, `ablation_h2.md`

3 seeds × 7 val streams, paired vs h2. Tuning (one parameter at a time): lr, teacher EMA, loss weights, BN momentum
range / gain, T3A K are all at their optimum or within noise (−0.33 … +0.07). Only the consistency augmentation
matters: `no_geom` +0.47 (17/21), `full` +0.67 (20/21) vs `soft`. Full tables: `analysis/tune_h2.md` (tuning),
`analysis/ablation_h2.md` (leave-one-out and cumulative). Leave-one-out losses: adaptive momentum −5.2, all extra
views −3.4, consistency −2.7, upscaled views −2.1, LAME −1.8, flip −0.85, learn first + extra steps −0.78,
extra steps −0.49, entropy −0.17, sqrt −0.16, temporal affinity −0.05, T3A −0.04.
Test files: `out/arbn_h2_nogeom/` (`config_h2_nogeom.json`), `out/arbn_h2_fullaug/` (`config_h2_fullaug.json`).

## Speed variants (time_exp.py, fast_h2.md)
Method-only time for 192 batches on the 4090 (harness adds about 4 s): h2 fullaug 38.8 s; without entropy loss 31.5 s;
without T3A 38.5 s; without temporal affinity 38.9 s; 1 extra step 35.1 s; 0 extra steps 31.2 s; without the flipped
224 view 37.3 s. Paired accuracy vs fullaug (3 seeds x 7 streams): - entropy -0.18 (7/21); - entropy with 3 / 4 extra
steps -0.14 / -0.10; 1 extra step -0.48; - entropy and 1 extra step -0.78.
`config_h2_fullaug_noent.json` -> `out/arbn_h2_fullaug_noent`: test 33.8-39.3 s over five runs (one earlier run 68.8 s,
cause not identified), harness val 0.9517 / 0.9621 / 0.9512.

## After public 0.93538 for h2_noent (2026-11-15)
Public: h2_noent 0.93538; two of fullaug / nogeom / fullaug_noent scored 0.93912 and 0.93066 (mapping not recorded).
All inside the h2 family band (0.9377-0.9443) except 0.93066. Stronger augmentation did not transfer to test.
Test stream structure (one harness pass, conv1 output statistics per batch): 12 segments (onsets at batches 0, 18,
36, 46, 62, 78, 92, 105, 118, 138, 156, 175), mostly distinct domains -> no use for per-domain state caching.
Augmentation groups (`aug_groups` 4 / 8 / 16, also in the extra steps; groups_h2.md): -0.11 / -0.16 / -0.02 -> no gain.
