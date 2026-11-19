# CVPDL 2026 HW2 — test-time adaptation

Online test-time adaptation of a source-trained ResNet-50 (160 px, 20 classes) on a continually shifting,
label-skewed stream (batch 64). Every method runs through the official online harness (`starter/harness.py`):
it sees each batch once, in stream order, and must answer before the next batch arrives.
Limits: test stream ≤ 90 s on an RTX 3090, ≤ 10 GB VRAM, no source data and no other weights.

## Final submission

`arbn_h2_up192f_176` (config `starter/config_h2_up192f_176.json`, run via `--method arbn_best`):

- **Adaptive Robust BN**: EMA of BN statistics whose momentum follows the shallow-layer symmetric KL
  (fast after a domain shift, slow inside a domain)
- **TENT** entropy loss on BN affine parameters + **EMA mean teacher** (0.999) with consistency on soft-augmented images
- **Learn first, then predict**, with 2 extra consistency steps on half the batch
- **8-view prediction**: original + whole image resized to 224 / 192 / 176, each with its horizontal flip (fp16)
- **T3A** classifier adjustment and **LAME** output smoothing with a temporal affinity term

Val 0.9541 (seed 0). Clean RTX 4090 test time 56.5 s, peak VRAM 4.4 GB.

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# put the homework data in data/ (val/, test/, sample_submission.csv) and source_resnet50_160.pth in the repo root
cd starter
ARBN_CONFIG=config_h2_up192f_176.json ../.venv/bin/python run.py --method arbn_best --split test \
    --data ../data --ckpt ../source_resnet50_160.pth --out ../out/arbn_h2_up192f_176
```

Use `--split val` to print the val accuracy instead of writing `test_pred.csv`.

## Layout

| Path | Content |
|---|---|
| `starter/harness.py`, `run.py`, `classes.txt` | course starter code, unmodified |
| `starter/methods.py` | method registry (baselines + `arbn*` variants) |
| `starter/tta/` | TTA methods: TENT, CoTTA, RoTTA, RMT, ROID, TRIBE, T3A, Robust BN, `arbn_teacher` |
| `starter/config_*.json` | `arbn_teacher` configurations for the experiments (pick one with `ARBN_CONFIG`) |
| `analysis/` | benchmark, ablation and proxy-stream scripts, logs and result tables — see `analysis/README.md` |
