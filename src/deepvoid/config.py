"""Run configuration.

A config is a nested dict, usually read from YAML. Anything left out falls
back to ``DEFAULTS``, and the fully resolved config is saved with each run,
so a run folder always records every setting that produced it. See
``docs/training.md`` for what each setting does.
"""

from __future__ import annotations

import copy
import re
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
        "arch": "unet",        # "unet" or "unetr" (vision transformer encoder, see models.unetr)
        "depth": 4,
        "base_filters": 16,
        "batch_norm": True,
        "bn_momentum": 0.99,
        "dropout": 0.0,
        "conditioning": None,
        "deep_supervision": False,
        "strip_pool": None,    # e.g. {axes: [z], length: 5}; see models.unet._strip_context
        # Read only when arch is "unetr"; depth is then set by the patch size.
        "vit": {"patch": 8, "dim": 240, "blocks": 9, "heads": 8, "mlp_ratio": 4},
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
        # Distillation: also learn a teacher run's class scores, see losses.DistillLoss.
        # e.g. {teacher: dm_smooth2ch_depth5, alpha: 0.5, temperature: 2.0}
        "distill": None,
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
    check_strip_pool(cfg["model"]["strip_pool"])
    check_arch(cfg)
    check_distill(cfg)
    return cfg


def check_distill(cfg: dict) -> None:
    """Validate ``training.distill``: None, or {teacher, alpha in [0, 1], temperature > 0}."""
    d = cfg["training"]["distill"]
    if d is None:
        return
    if not isinstance(d, dict) or not d.get("teacher") or set(d) - {"teacher", "alpha",
                                                                     "temperature"}:
        raise ValueError("training.distill must be null or {teacher: <run>, alpha: a, "
                         "temperature: T}")
    d.setdefault("alpha", 0.5)
    d.setdefault("temperature", 2.0)
    if not 0.0 <= float(d["alpha"]) <= 1.0:
        raise ValueError("training.distill.alpha must be between 0 and 1")
    if not float(d["temperature"]) > 0:
        raise ValueError("training.distill.temperature must be positive")
    if cfg["model"]["deep_supervision"]:
        raise ValueError("training.distill does not work with model.deep_supervision yet")
    if cfg["data"]["subcubes"] is not None:
        raise ValueError("training.distill does not work with data.subcubes")


def check_arch(cfg: dict) -> None:
    """Validate ``model.arch`` and, for "unetr", the crop sizes and the U-Net-only options.

    Every crop side must be a multiple of the patch size. The prediction crop
    may not exceed the training crop: the position embedding is defined for
    any grid, but the model has only learned to use the token positions of
    the training crop.
    """
    model, arch = cfg["model"], cfg["model"]["arch"]
    if arch not in ("unet", "unetr"):
        raise ValueError("model.arch must be 'unet' or 'unetr'")
    if arch == "unet":
        return
    for key in ("conditioning", "strip_pool"):
        if model[key] is not None:
            raise ValueError(f"model.{key} is not available with model.arch: unetr")
    if model["deep_supervision"]:
        raise ValueError("model.deep_supervision is not available with model.arch: unetr")
    patch = model["vit"]["patch"]
    tr, ev = cfg["training"], cfg["evaluate"]
    for name, side in (("training.crop", tr["crop"]), ("training.val_crop", tr["val_crop"]),
                       ("evaluate.crop", ev["crop"])):
        if side % patch:
            raise ValueError(f"{name} ({side}) must be a multiple of model.vit.patch ({patch})")
    if ev["crop"] > tr["crop"]:
        raise ValueError(f"evaluate.crop ({ev['crop']}) must not exceed training.crop "
                         f"({tr['crop']}) with model.arch: unetr")


def check_strip_pool(strip: dict | None) -> None:
    """Validate ``model.strip_pool``: None, or {axes: [x|y|z, ...], length: odd int >= 3}."""
    if strip is None:
        return
    if not isinstance(strip, dict) or set(strip) != {"axes", "length"}:
        raise ValueError("model.strip_pool must be null or {axes: [...], length: n}")
    axes = strip["axes"]
    if not axes or len(set(axes)) != len(axes) or not set(axes) <= {"x", "y", "z"}:
        raise ValueError("model.strip_pool.axes must be distinct axes from x, y, z")
    n = strip["length"]
    if not isinstance(n, int) or isinstance(n, bool) or n < 3 or n % 2 == 0:
        raise ValueError("model.strip_pool.length must be an odd integer >= 3 "
                         "(bottleneck cells, centered on each voxel)")


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
    """Refuse augmentation or splits that would mix up a redshift-space line of sight.

    Inputs marked ``rsd: true`` in the catalog are stretched along their
    line of sight (``los_axis``, default 2). Rotating them freely would put
    that stretch along other axes, which no real survey shows, so they need
    ``training.rotate: z`` (or no rotation). An input with its line of sight
    along x or y is turned so it lies along z (see :func:`los_swap`), which
    the split must allow.
    """
    rsd = [n for n in cfg["data"]["inputs"] if catalog.info(n).get("rsd")]
    if not rsd:
        return
    if cfg["training"]["rotate"] is True:
        raise ValueError(f"{rsd} are in redshift space; set training.rotate to 'z' so "
                         "rotations keep the line of sight along z")
    for name in rsd:
        los_swap(cfg, catalog, name)


def los_swap(cfg: dict, catalog, name: str) -> int | None:
    """The axis to swap with z so input ``name`` has its line of sight along z.

    None for real-space inputs and for a line of sight already along z.
    For a line of sight along x or y, the input and its labels are swapped
    with z (``observe.los_to_z``) for training and scoring. Refused when the
    line of sight is the split axis (tracers would move between the
    training, validation and test slabs) or when the swap would move the
    split axis (the slabs would no longer line up).
    """
    info = catalog.info(name)
    if not info.get("rsd"):
        return None
    los = info.get("los_axis", 2)
    split = cfg["data"]["split"]["axis"]
    if los == split:
        raise ValueError(f"{name} has its line of sight along the split axis "
                         f"({'xyz'[split]}): tracers would move between the training, "
                         "validation and test slabs")
    if los == 2:
        return None
    if split == 2:
        raise ValueError(f"{name}: turning its line of sight ({'xyz'[los]}) onto z would "
                         "move the split axis (z)")
    return los


def input_label(name: str) -> str:
    """Short label for a dataset name, used in metric names and plot titles.

    'tng300-3/subhalos/L10' -> 'L10'; a variant such as
    'tng300-3/subhalos/L10/rsd-z' keeps its spacing: 'L10_rsd-z'.
    """
    parts = name.split("/")
    if len(parts) > 1 and not re.fullmatch(r"L[\d.]+", parts[-1]) and re.fullmatch(r"L[\d.]+", parts[-2]):
        return f"{parts[-2]}_{parts[-1]}"
    return parts[-1]


def check_unique_labels(names: list[str]) -> None:
    """Refuse inputs whose short labels collide, so no metric overwrites another."""
    labels = [input_label(n) for n in names]
    dupes = sorted({l for l in labels if labels.count(l) > 1})
    if dupes:
        raise ValueError(f"inputs share the labels {dupes}; rename them in the catalog")

