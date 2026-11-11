"""Train and evaluate a model from a config.

    python -m deepvoid.train configs/tng_dm_baseline.yaml
    python -m deepvoid.train configs/tng_dm_baseline.yaml --seeds 1 2 3

or in a notebook, ``run("configs/tng_dm_baseline.yaml")``.

Each run gets its own folder, ``<output>/<name>`` (``<name>_seed<n>`` when
seeds are given), holding:

- ``config.yaml``: every setting used, defaults included
- ``normalization.json``, ``split.json``, ``class_weights.json``
- ``log.csv``: one row per epoch, rewritten in full each epoch
- ``tensorboard/``: losses, scores and a validation slice per epoch
- ``best.keras``: the model at the best validation score
- ``last.keras``, ``ema.npz``, ``state.json``: what a resumed run needs
- ``test_scores.json``: scores on the held-out test slab, written at the end

Running the same config again continues an interrupted run from its last
completed epoch (set ``training.resume: false`` to refuse instead). See
``docs/training.md`` for the settings. Requires TensorFlow.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import keras
import numpy as np

from .callbacks import EMA, RunState, Validation, read_log
from .catalog import Catalog
from .config import load_config, save_config
from .evaluate import confusion_matrix, scores
from .losses import build_loss
from .models import unet3d
from .pipeline import CropSampler
from .predict import predict_volume
from .preprocess import Normalizer, auto_class_weights
from .split import Region, slab_split


def run(config, seed: int | None = None, catalog: Catalog | None = None) -> Path:
    """Train (or resume) one run, score it on the test slab, return its folder."""
    cfg = load_config(config)
    if seed is not None:
        cfg["seed"] = seed
        cfg["name"] = f"{cfg['name']}_seed{seed}"
    tr = cfg["training"]
    catalog = catalog or Catalog()
    out = Path(cfg["output"]) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)

    resuming = tr["resume"] and (out / "state.json").exists() and (out / "last.keras").exists()
    if resuming:
        _check_same_config(cfg, out / "config.yaml")
        print(f"resuming {out}")
    elif (out / "state.json").exists():
        raise FileExistsError(f"{out} already has a run; use a new name or resume: true")
    save_config(cfg, out / "config.yaml")

    keras.utils.set_random_seed(cfg["seed"])
    if tr["mixed_precision"]:
        keras.mixed_precision.set_global_policy("mixed_float16")

    # Data, split and normalization (fit on the training slab only).
    labels = catalog.load(cfg["data"]["labels"])
    regions = slab_split(labels.n, **cfg["data"]["split"])
    train_mask = regions["train"].mask(labels.n)
    inputs = _normalized_inputs(cfg, catalog, train_mask, out, resuming)
    (out / "split.json").write_text(json.dumps({k: vars(v) for k, v in regions.items()}))

    class_weights = tr["loss"]["class_weights"]
    if class_weights == "auto":
        class_weights = auto_class_weights(labels.data[train_mask])
    (out / "class_weights.json").write_text(json.dumps(class_weights))

    y = [labels.data] * len(inputs)
    sampling = dict(crop=tr["crop"], batch_size=tr["batch_size"])
    train_data = CropSampler(inputs, y, regions["train"], steps_per_epoch=tr["steps_per_epoch"],
                             rotate=tr["rotate"], seed=cfg["seed"], **sampling)
    val_source = CropSampler(inputs, y, regions["val"], steps_per_epoch=tr["val_batches"],
                             rotate=False, seed=cfg["seed"] + 1, **sampling)
    val_batches = [(np.asarray(x), np.asarray(t, np.uint8))
                   for x, t in (val_source[i] for i in range(len(val_source)))]
    del val_source

    # Model and optimizer, fresh or restored.
    loss = build_loss(tr["loss"], class_weights)
    if resuming:
        model = keras.models.load_model(out / "last.keras")
        optimizer = model.optimizer
    else:
        model = unet3d(**cfg["model"])
        optimizer = _optimizer(tr)
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
                 ema=ema, tb_local=tb_local, state=state, rows=read_log(out / "log.csv")),
    ]
    if first_epoch < tr["epochs"] and state.get("wait", 0) < patience:
        model.fit(train_data, epochs=tr["epochs"], initial_epoch=first_epoch,
                  callbacks=callbacks, verbose=2)
    else:
        print("training already finished")

    evaluate_run(out, catalog)
    return out


def evaluate_run(out: str | Path, catalog: Catalog | None = None) -> dict:
    """Predict each input's full box with ``best.keras`` and score the test slab.

    Writes ``test_scores.json`` (scores and confusion matrix per input) and
    returns the same dict.
    """
    out = Path(out)
    cfg = load_config(out / "config.yaml")
    ev = cfg["evaluate"]
    catalog = catalog or Catalog()
    model = keras.models.load_model(out / "best.keras")
    norms = json.loads((out / "normalization.json").read_text())
    labels = catalog.load(cfg["data"]["labels"]).data
    test = Region(**json.loads((out / "split.json").read_text())["test"])
    mask = test.mask(labels.shape[0])

    results = {}
    for name in cfg["data"]["inputs"]:
        x = Normalizer(**norms[name])(catalog.load(name).data)
        probs = predict_volume(lambda b: model.predict_on_batch(b), x, crop=ev["crop"],
                               margin=ev["margin"], batch_size=ev["batch_size"], tta=ev["tta"])
        cm = confusion_matrix(labels, probs.argmax(-1), mask=mask)
        results[name] = {**scores(cm), "confusion_matrix": cm.tolist()}
        print(name, {k: round(v, 3) for k, v in results[name].items()
                     if k in ("mcc", "void_f1", "halo_f1", "balanced_accuracy")})
    (out / "test_scores.json").write_text(json.dumps(results, indent=2))
    return results


def run_seeds(config, seeds: list[int], catalog: Catalog | None = None) -> dict:
    """Run one config with several seeds; return mean and std of each test score."""
    outs = [run(config, seed=s, catalog=catalog) for s in seeds]
    per_seed = [json.loads((o / "test_scores.json").read_text()) for o in outs]
    summary = {}
    for name in per_seed[0]:
        keys = [k for k in per_seed[0][name] if k != "confusion_matrix"]
        values = {k: np.array([r[name][k] for r in per_seed]) for k in keys}
        summary[name] = {k: {"mean": float(v.mean()),
                             "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
                         for k, v in values.items()}
    return summary


def _normalized_inputs(cfg, catalog, train_mask, out, resuming):
    path = out / "normalization.json"
    saved = json.loads(path.read_text()) if resuming and path.exists() else {}
    inputs, norms = [], {}
    for name in cfg["data"]["inputs"]:
        data = catalog.load(name).data
        norms[name] = Normalizer(**saved[name]) if name in saved else Normalizer.fit(data, train_mask)
        inputs.append(norms[name](data))
    path.write_text(json.dumps({k: v.to_dict() for k, v in norms.items()}, indent=2))
    return inputs


def _optimizer(tr):
    opt, sched = tr["optimizer"], tr["schedule"]
    lr = opt["learning_rate"]
    if sched["name"] == "cosine":
        total = tr["epochs"] * tr["steps_per_epoch"]
        warmup = sched["warmup_epochs"] * tr["steps_per_epoch"]
        lr = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.0, decay_steps=max(total - warmup, 1),
            alpha=sched["min_lr_ratio"], warmup_target=opt["learning_rate"],
            warmup_steps=warmup)
    elif sched["name"] != "constant":
        raise ValueError("schedule must be 'cosine' or 'constant'")
    if opt["name"] == "adamw":
        return keras.optimizers.AdamW(learning_rate=lr, weight_decay=opt["weight_decay"],
                                      clipnorm=opt["clipnorm"])
    if opt["name"] == "adam":
        return keras.optimizers.Adam(learning_rate=lr, clipnorm=opt["clipnorm"])
    raise ValueError("optimizer must be 'adamw' or 'adam'")


def _check_same_config(cfg, saved_path):
    """Refuse to resume with different settings, except the epoch count and evaluation."""
    saved = load_config(saved_path)
    a, b = json.loads(json.dumps(cfg)), json.loads(json.dumps(saved))
    for c in (a, b):
        c["training"].pop("epochs")
        c["training"].pop("resume")
        c.pop("evaluate")
    if a != b:
        raise ValueError(f"settings differ from the run being resumed in {saved_path.parent}; "
                         "use a new name for a new run")


def main():
    parser = argparse.ArgumentParser(description="Train a DeepVoid model from a config.")
    parser.add_argument("config")
    parser.add_argument("--seeds", type=int, nargs="*")
    args = parser.parse_args()
    if args.seeds:
        print(json.dumps(run_seeds(args.config, args.seeds), indent=2))
    else:
        print(run(args.config))


if __name__ == "__main__":
    main()
