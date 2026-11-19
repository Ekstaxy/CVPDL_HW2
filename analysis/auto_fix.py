"""Follow-up to auto_explore: fit the best view set into the time budget (cheaper view set, or fp16 views)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import auto_explore as AE  # noqa: E402
base = AE.evaluate({})
ref = AE.evaluate({"flip": True, "upscales": [224], "upscale_flip": True, "aug": "soft"})
C = {"soft_U224": {"flip": True, "upscales": [224], "upscale_flip": False, "aug": "soft"},
     "soft_U192": {"flip": True, "upscales": [192], "upscale_flip": False, "aug": "soft"},
     "soft_U224UF_amp": {"flip": True, "upscales": [224], "upscale_flip": True, "aug": "soft", "view_amp": True},
     "soft_U224_amp": {"flip": True, "upscales": [224], "upscale_flip": False, "aug": "soft", "view_amp": True}}
out = {}
for name, cfg in C.items():
    rows = AE.evaluate(cfg); sm = AE.summarize(rows)
    dm, dmin, w = AE.paired(rows, base); dr, drmin, wr = AE.paired(rows, ref)
    res, tp = AE.harness_test_seconds(cfg, name)
    healthy, _ = AE.health(tp, AE.ROOT / "out/arbn_teacher_aug_no_geom/test_pred.csv")
    out[name] = dict(cfg=cfg, **{k: round(float(v), 4) for k, v in sm.items()}, vs_default=round(dm, 4), wins=w,
                     vs_4view=round(dr, 4), harness_val=res["val"]["acc"], test_s=res["test"]["seconds"],
                     vram=res["test"]["peak_vram_mib"], healthy=healthy, file=str(tp.relative_to(AE.ROOT)))
    AE.log("auto_fix", name, out[name])
json.dump(out, open(AE.AUTO / "auto_fix.json", "w"), indent=1)
