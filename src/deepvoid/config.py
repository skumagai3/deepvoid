"""Run configuration.

A config is a nested dict, usually read from YAML. Anything left out falls
back to ``DEFAULTS``, and the fully resolved config is saved with each run,
so a run folder always records every setting that produced it. See
``docs/training.md`` for what each setting does.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "name": None,
    "output": "runs",
    "catalog": None,       # path of a catalog.yaml; None uses the packaged catalog
    "seed": 12,
    "data": {
        "inputs": [],
        "labels": None,
        "split": {"fractions": [0.7, 0.15, 0.15], "buffer": 16, "axis": 0},
        "smoothing": None,
        "normalization": "log_standard",
        "subcubes": None,
    },
    "model": {
        "depth": 4,
        "base_filters": 16,
        "batch_norm": True,
        "bn_momentum": 0.99,
        "dropout": 0.0,
        "conditioning": None,
        "deep_supervision": False,
    },
    "training": {
        "crop": 96,
        "batch_size": 8,
        "steps_per_epoch": 200,
        "val_crop": 64,
        "val_batches": 32,
        "epochs": 80,
        "rotate": True,
        "init_from": None,
        "curriculum": None,
        "optimizer": {"name": "adamw", "learning_rate": 1.0e-3,
                      "weight_decay": 1.0e-4, "clipnorm": 1.0},
        "schedule": {"name": "cosine", "warmup_epochs": 3, "min_lr_ratio": 0.01,
                     # read only by the "plateau" schedule (data.subcubes runs)
                     "factor": 0.25, "patience": 10, "min_lr": 1.0e-7},
        "ema": 0.999,
        "loss": {"name": "scce", "class_weights": None, "dice_weight": 1.0, "void_weight": 0.0},
        "early_stopping": {"monitor": "val_void_mcc", "patience": 15, "min_delta": 0.002},
        "mixed_precision": True,
        "xla": True,
        "resume": True,
    },
    "evaluate": {"crop": 128, "margin": 32, "batch_size": 4, "tta": "none", "inputs": None,
                 "labels": None, "void_threshold": None, "void_objective": "mcc"},
}


def load_config(source: str | Path | dict) -> dict[str, Any]:
    """Read a config from a YAML path or a dict and fill in the defaults."""
    if isinstance(source, dict):
        user = copy.deepcopy(source)
    else:
        with open(source) as f:
            user = yaml.safe_load(f) or {}
    cfg = merge(DEFAULTS, user)
    if not cfg["name"]:
        raise ValueError("the config needs a 'name'")
    if not cfg["data"]["inputs"] or not cfg["data"]["labels"]:
        raise ValueError("the config needs data.inputs and data.labels")
    if cfg["training"]["rotate"] not in (True, False, "z"):
        raise ValueError("training.rotate must be true, false or 'z'")
    return cfg


def merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``.

    Unknown keys are refused, so a typo in a config fails loudly instead of
    being ignored.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key not in base:
            raise KeyError(f"unknown config setting {key!r}")
        if isinstance(base[key], dict) and isinstance(value, dict):
            out[key] = merge(base[key], value)
        else:
            out[key] = value
    return out


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def check_line_of_sight(cfg: dict, catalog) -> None:
    """Refuse augmentation that would turn a redshift-space line of sight.

    Inputs marked ``rsd: true`` in the catalog are stretched along their
    line of sight (``los_axis``, default 2). Rotating them would put that
    stretch along other axes, which no real survey shows, so they need
    ``training.rotate: z`` (or no rotation) and a line of sight along z.
    """
    rsd = [n for n in cfg["data"]["inputs"] if catalog.info(n).get("rsd")]
    if not rsd:
        return
    if cfg["training"]["rotate"] is True:
        raise ValueError(f"{rsd} are in redshift space; set training.rotate to 'z' so "
                         "rotations keep the line of sight along z")
    axes = {catalog.info(n).get("los_axis", 2) for n in rsd}
    if cfg["training"]["rotate"] == "z" and axes != {2}:
        raise ValueError(f"rotate 'z' needs the line of sight along z; these inputs have "
                         f"los_axis {sorted(axes)}")

