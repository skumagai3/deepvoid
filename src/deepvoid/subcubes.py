"""The first paper's data protocol, for measuring what it does to the scores.

The first DeepVoid paper cut the 512^3 box into 4^3 = 64 non-overlapping
128^3 subcubes, stored each with copies rotated by 90, 180 and 270 degrees
about one axis (``np.rot90``, axes 0 and 1), and split all 256 examples
80/20 at random into training and validation (sklearn ``train_test_split``,
``random_state=12``). The rotated copies of one region could land on both
sides, and the validation set was used both to choose the model and to
report its scores.

This module reproduces that protocol, with two changes needed to measure
it:

- a layer of subcubes (``holdout``, by default x in [384, 512)) is held out
  entirely and used only to score the final model, so every run has an
  unseen test region;
- ``split_by`` chooses how the remaining examples are split:
  ``"copy"`` treats each rotated copy as its own example (the paper's way),
  ``"region"`` keeps the rotations of a region together (the control).

Comparing the validation and test scores of a ``"copy"`` run with those of
a ``"region"`` run measures how much the paper's split raises the
validation score.

Training draws whole subcubes, shuffled each epoch, with no augmentation
beyond the stored rotations, as in the paper. Optimizer, schedule, loss,
weight averaging, early stopping and normalization are set in the config;
``configs/papersplit`` sets them to the paper's (Adam, learning rate cut on a
validation-loss plateau, no weight averaging, min-max scaling).

Enable with ``data.subcubes`` in a config, e.g.::

    data:
      subcubes: {size: 128, rotations: 4, val_fraction: 0.2,
                 split_by: copy, holdout: [384, 512], seed: 12}
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np

SETTINGS = {"size": 128, "rotations": 4, "val_fraction": 0.2, "split_by": "copy",
            "holdout": [384, 512], "seed": 12}


def settings(user: dict) -> dict:
    """Fill in defaults for ``data.subcubes`` and refuse unknown keys."""
    unknown = set(user) - set(SETTINGS)
    if unknown:
        raise KeyError(f"unknown data.subcubes setting(s): {sorted(unknown)}")
    s = {**SETTINGS, **user}
    if s["split_by"] not in ("copy", "region"):
        raise ValueError("data.subcubes.split_by must be 'copy' or 'region'")
    if not 1 <= s["rotations"] <= 4:
        raise ValueError("data.subcubes.rotations must be 1 to 4")
    return s


def subcube_split(n: int, size: int = 128, rotations: int = 4, val_fraction: float = 0.2,
                  split_by: str = "copy", holdout=(384, 512), seed: int = 12) -> dict:
    """Training, validation and test examples as (x, y, z, k) tuples.

    (x, y, z) is the subcube origin in cells and k the number of 90-degree
    rotations about the first two axes. Subcubes lie on a grid with spacing
    ``size`` and do not overlap. Those inside ``holdout`` (an x range) are the
    test set, unrotated; the others, with ``rotations`` copies each, are split
    into training and validation. The number of validation examples (or
    regions) is ceil(val_fraction * count), as in sklearn.
    """
    lo, hi = holdout
    starts = range(0, n - size + 1, size)
    test_x = [x for x in starts if x >= lo and x + size <= hi]
    fit_x = [x for x in starts if x + size <= lo or x >= hi]
    regions = [(x, y, z) for x in fit_x for y in starts for z in starts]
    test = [(x, y, z, 0) for x in test_x for y in starts for z in starts]
    rng = np.random.default_rng(seed)
    if split_by == "copy":
        examples = [r + (k,) for r in regions for k in range(rotations)]
        order = rng.permutation(len(examples))
        n_val = math.ceil(val_fraction * len(examples))
        val = [examples[i] for i in sorted(order[:n_val])]
        train = [examples[i] for i in sorted(order[n_val:])]
    elif split_by == "region":
        order = rng.permutation(len(regions))
        n_val = math.ceil(val_fraction * len(regions))
        val_r = {regions[i] for i in order[:n_val]}
        val = [r + (k,) for r in regions if r in val_r for k in range(rotations)]
        train = [r + (k,) for r in regions if r not in val_r for k in range(rotations)]
    else:
        raise ValueError("split_by must be 'copy' or 'region'")
    return {"train": train, "val": val, "test": test}


def copy_fraction(split: dict) -> float:
    """Fraction of validation examples whose region also has a copy in training."""
    seen = {e[:3] for e in split["train"]}
    val = split["val"]
    return sum(e[:3] in seen for e in val) / len(val) if val else 0.0


def extract(volume: np.ndarray, example: tuple, size: int) -> np.ndarray:
    """The subcube for one example, rotated k times by 90 degrees (axes 0, 1)."""
    x, y, z, k = example
    cube = volume[x:x + size, y:y + size, z:z + size]
    return np.rot90(cube, k, axes=(0, 1))


def fit_mask(n: int, holdout) -> np.ndarray:
    """Voxels outside the held-out x range, used for normalization and class weights."""
    mask = np.ones((n, n, n), bool)
    mask[holdout[0]:holdout[1]] = False
    return mask


def run_subcubes(cfg: dict, catalog, out: Path, resuming: bool) -> Path:
    """Train and score one run under the subcube protocol (called from ``train.run``)."""
    import keras

    from .callbacks import EMA, RunState, Validation, read_log
    from .inputs import n_channels
    from .losses import build_loss
    from .models import unet3d
    from .preprocess import auto_class_weights
    from .train import _normalized_inputs, _optimizer, _label

    sc = settings(cfg["data"]["subcubes"])
    tr = cfg["training"]
    if len(cfg["data"]["inputs"]) != 1:
        raise ValueError("the subcube protocol takes one input, as in the paper")
    if cfg["model"]["deep_supervision"] or cfg["model"]["conditioning"]:
        raise ValueError("the subcube protocol does not support deep supervision or conditioning")

    labels = catalog.load(cfg["data"]["labels"]).data
    n = labels.shape[0]
    split = subcube_split(n, **sc)
    mask = fit_mask(n, sc["holdout"])
    x = _normalized_inputs(cfg, catalog, mask, out, resuming)[0]
    info = {"settings": sc, "counts": {k: len(v) for k, v in split.items()},
            "validation_with_a_copy_in_training": copy_fraction(split),
            **{k: [list(map(int, e)) for e in v] for k, v in split.items()}}
    (out / "split.json").write_text(json.dumps(info))
    print(f"subcube split ({sc['split_by']}): {info['counts']}; "
          f"{info['validation_with_a_copy_in_training']:.1%} of validation examples "
          "have a rotated copy of their region in training")

    class_weights = tr["loss"]["class_weights"]
    if class_weights == "auto":
        class_weights = auto_class_weights(labels[mask])
    (out / "class_weights.json").write_text(json.dumps(class_weights))

    size, bs = sc["size"], tr["batch_size"]
    train_data = SubcubeSampler(x, labels, split["train"], size, bs, seed=cfg["seed"])
    val_batches = {_label(cfg["data"]["inputs"][0]): _batches(x, labels, split["val"], size, bs)}

    loss = build_loss(tr["loss"], class_weights)
    if resuming:
        model = keras.models.load_model(out / "last.keras")
        optimizer = model.optimizer
    else:
        model = unet3d(**cfg["model"], in_channels=n_channels(cfg["data"]["smoothing"]))
        optimizer = _optimizer({**tr, "steps_per_epoch": len(train_data)})
    model.compile(optimizer=optimizer, loss=loss, metrics=["sparse_categorical_accuracy"],
                  jit_compile=tr["xla"])

    state = json.loads((out / "state.json").read_text()) if resuming else {}
    first_epoch = state.get("epoch", -1) + 1
    patience = tr["early_stopping"]["patience"]
    tb_local = Path(tempfile.gettempdir()) / "deepvoid_tensorboard" / cfg["name"]
    if resuming and (out / "tensorboard").exists() and not tb_local.exists():
        shutil.copytree(out / "tensorboard", tb_local)
    ema = EMA(tr["ema"]) if tr["ema"] else None
    if ema and resuming and (out / "ema.npz").exists():
        ema.load(out / "ema.npz")
    callbacks = ([ema] if ema else []) + [
        Validation(val_batches, tb_local, ema),
        keras.callbacks.TensorBoard(str(tb_local), write_graph=False),
        RunState(out, train_data, monitor=tr["early_stopping"]["monitor"], patience=patience,
                 min_delta=tr["early_stopping"]["min_delta"], ema=ema, tb_local=tb_local,
                 state=state, rows=read_log(out / "log.csv")),
    ]
    if tr["schedule"]["name"] == "plateau":
        # The paper's schedule: multiply the learning rate by ``factor`` when
        # the validation loss has not improved for ``patience`` epochs.
        sched = tr["schedule"]
        callbacks.insert(-1, keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=sched["factor"], patience=sched["patience"],
            min_lr=sched["min_lr"], verbose=1))
    if first_epoch < tr["epochs"] and state.get("wait", 0) < patience:
        model.fit(train_data, epochs=tr["epochs"], initial_epoch=first_epoch,
                  callbacks=callbacks, verbose=2)
    else:
        print("training already finished")
    evaluate_subcubes(out, cfg, catalog)
    return out


def evaluate_subcubes(out: Path, cfg: dict, catalog) -> dict:
    """Score best.keras on its validation examples and on the held-out test subcubes.

    Each 128^3 example is predicted whole, as in the paper (no tiling or
    margin), and the predicted class is the argmax. Writes
    ``test_scores.json`` with ``validation`` and ``test`` entries.
    """
    from .evaluate import confusion_matrix, scores
    from .inputs import build as build_input
    from .models import load_model
    from .report import plain
    from .train import _label

    info = json.loads((out / "split.json").read_text())
    sc = info["settings"]
    norms = json.loads((out / "normalization.json").read_text())
    labels = catalog.load(cfg["data"]["labels"]).data
    name = cfg["data"]["inputs"][0]
    x = build_input(catalog, name, cfg["data"]["smoothing"], norms,
                    fit_mask(labels.shape[0], sc["holdout"]))
    model = load_model(out / "best.keras")
    result = {}
    for key, part in (("validation", "val"), ("test", "test")):
        cm, by_region = 0, {}
        examples = [tuple(e) for e in info[part]]
        batches = _batches(x, labels, examples, sc["size"], cfg["evaluate"]["batch_size"])
        done = 0
        for xb, yb in batches:
            pred = np.asarray(model.predict_on_batch(xb)).argmax(-1)
            for y_one, p_one in zip(yb, pred):
                one = confusion_matrix(y_one, p_one)
                region = examples[done][:3]
                by_region[region] = by_region.get(region, 0) + one
                cm = cm + one
                done += 1
        result[key] = {**scores(cm), "confusion_matrix": np.asarray(cm).tolist(),
                       "examples": len(examples), **region_summary(by_region)}
    result["validation_with_a_copy_in_training"] = info["validation_with_a_copy_in_training"]
    scored = {_label(name): plain(result)}
    (out / "test_scores.json").write_text(json.dumps(scored, indent=2))
    for key in ("validation", "test"):
        r = result[key]
        m = r["by_region"]["mean"]
        e = r["by_region"]["standard_error"]
        print(f"{key}: mcc {r['mcc']:.3f}, void_mcc {r['void_mcc']:.3f}, "
              f"void_f1 {r['void_f1']:.3f} ({r['examples']} subcubes). "
              f"Per region ({r['by_region']['regions']} regions): "
              f"mcc {m['mcc']:.3f} ± {e['mcc']:.3f}, void_mcc {m['void_mcc']:.3f} ± {e['void_mcc']:.3f}")
    return scored


REGION_SCORES = ("mcc", "void_mcc", "void_f1")


def region_summary(by_region: dict) -> dict:
    """Scores per region, with their mean and standard error.

    ``by_region`` maps a region's origin (x, y, z) to its confusion matrix,
    summed over whichever rotated copies of that region are in the set.
    Rotated copies of one region are not independent samples, so the spread
    is taken over regions: the standard error is the standard deviation of
    the per-region scores (ddof = 1) divided by sqrt(number of regions).
    """
    from .evaluate import scores

    regions = sorted(by_region)
    rows = [scores(np.asarray(by_region[r])) for r in regions]
    values = {k: [float(row[k]) for row in rows] for k in REGION_SCORES}
    n = len(regions)
    mean = {k: float(np.mean(v)) for k, v in values.items()}
    err = {k: float(np.std(v, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
           for k, v in values.items()}
    return {"by_region": {"regions": n, "mean": mean, "standard_error": err,
                          "origins": [list(r) for r in regions], "scores": values}}


def _with_channels(a: np.ndarray) -> np.ndarray:
    return a if a.ndim == 4 else a[..., None]


def _batches(x, labels, examples, size, batch_size):
    """Fixed (inputs, labels) batches for a list of examples."""
    out = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i:i + batch_size]
        xb = np.stack([_with_channels(extract(x, e, size)) for e in chunk]).astype(np.float32)
        yb = np.stack([extract(labels, e, size) for e in chunk]).astype(np.uint8)
        out.append((xb, yb))
    return out


try:
    import keras as _keras

    class SubcubeSampler(_keras.utils.PyDataset):
        """Whole training subcubes in a new random order each epoch."""

        def __init__(self, x, labels, examples, size, batch_size, seed=0, **kwargs):
            super().__init__(**kwargs)
            self.x, self.labels, self.examples = x, labels, list(examples)
            self.size, self.batch_size, self.seed = size, batch_size, seed
            self.set_epoch(0)

        def __len__(self) -> int:
            return len(self.examples) // self.batch_size

        def set_epoch(self, epoch: int) -> None:
            self.order = np.random.default_rng([self.seed, epoch]).permutation(len(self.examples))

        def __getitem__(self, index):
            idx = self.order[index * self.batch_size:(index + 1) * self.batch_size]
            chunk = [self.examples[i] for i in idx]
            xb = np.stack([_with_channels(extract(self.x, e, self.size)) for e in chunk])
            yb = np.stack([extract(self.labels, e, self.size) for e in chunk])
            return xb.astype(np.float32), yb.astype(np.int32)
except ImportError:                               # numpy-only use (tests, split inspection)
    SubcubeSampler = None
