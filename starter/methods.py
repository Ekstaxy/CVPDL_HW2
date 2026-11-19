"""TTA methods. Add your own here and register it in METHODS.

A method is a callable: method(x) -> LongTensor of labels in {0..19} for the normalized batch x.
It may update its own state (BN statistics, parameters, memory, ...) as the stream goes on.

Every method below follows its paper's original (ImageNet-C / reference-repo) setting; see the docstrings
in tta/*.py for hyper-parameters and the (rule-forced) deviations.
"""
import torch
import torch.nn as nn

from tta.arbn_teacher import AdaptiveRBNTeacher
from tta.arbn_tent import AdaptiveRBNLame, AdaptiveRBNTent
from tta.cotta import CoTTA
from tta.norm import AdaptiveRobustBN, BNAdapt, RobustBN
from tta.rmt import RMT
from tta.roid import ROID
from tta.rotta import RoTTA
from tta.tent import Tent
from tta.tribe import TRIBE


class Source:
    """No adaptation."""

    def __init__(self, model):
        self.model = model.eval()
        self.last_logits = None

    @torch.no_grad()
    def __call__(self, x):
        out = self.model(x)
        self.last_logits = out
        return out.argmax(1)


def ArbnBest(model):
    """arbn_teacher with the configuration in best_config.json (written by analysis/auto_explore.py).
    The file can be overridden with the environment variable ARBN_CONFIG."""
    import json
    import os
    path = os.environ.get("ARBN_CONFIG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_config.json"))
    cfg = json.load(open(path)) if os.path.exists(path) else {}
    for k in ("upscales",):
        if k in cfg and cfg[k] is not None:
            cfg[k] = tuple(cfg[k])
    return AdaptiveRBNTeacher(model, **cfg)


METHODS = {
    "source": Source,
    "bn_adapt": BNAdapt,
    "tent": Tent,
    "rbn": RobustBN,
    "arbn": AdaptiveRobustBN,
    "arbn_tent": AdaptiveRBNTent,
    "arbn_lame": AdaptiveRBNLame,
    "arbn_teacher": AdaptiveRBNTeacher,
    "arbn_best": ArbnBest,
    "rotta": RoTTA,
    "cotta": CoTTA,
    "rmt": RMT,
    "roid": ROID,
    "tribe": TRIBE,
}
