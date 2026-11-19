"""Within-segment accuracy curve on the test-like proxy (base config): is the error in the first batches after a
shift (adaptation speed) or spread over the whole segment (adaptation capacity)?
  .venv/bin/python analysis/testlike_curve.py <config json> [cfg override json]
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import testlike as TL  # noqa: E402
from tta.arbn_teacher import AdaptiveRBNTeacher  # noqa: E402
X, y, seg = TL.load()
c = json.load(open(sys.argv[1])); c.update(json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
c["upscales"] = tuple(c.get("upscales", ()))
ok = np.mean([TL.run(lambda m: AdaptiveRBNTeacher(m, **c), X, s)[0] == y for s in (0, 1)], 0)
b = ok.reshape(-1, 64).mean(1); s = 0
print("segment       first3  b4-8   rest   all")
for k, nb in enumerate(TL.SEG_B):
    q = b[s:s + nb]
    print(f"{TL.KINDS[k]:<12} {q[:3].mean():.3f}  {q[3:8].mean():.3f}  {q[8:].mean():.3f}  {q.mean():.3f}")
    s += nb
np.save(Path(__file__).resolve().parent / "testlike_curve_ok.npy", ok)
