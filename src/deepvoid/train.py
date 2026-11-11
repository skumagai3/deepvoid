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
  (one entry per input scored)
- ``test_figure.png``: a slice through the test slab and the confusion matrix
  for each input scored

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
from .inputs import build as build_input
from .inputs import n_channels
from .losses import build_loss
from .models import copy_weights, is_conditioned, load_model, unet3d
from .pipeline import CropSampler
from .predict import predict_volume
from .resample import upsampled_argmax
from .preprocess import auto_class_weights
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

    names = cfg["data"]["inputs"]
    conditioned = cfg["model"]["conditioning"] is not None
    spacings = [_spacing(catalog, n) for n in names] if conditioned else None
    y = [labels.data] * len(inputs)
    train_data = CropSampler(inputs, y, regions["train"], crop=tr["crop"],
                             batch_size=tr["batch_size"], steps_per_epoch=tr["steps_per_epoch"],
                             rotate=tr["rotate"], seed=cfg["seed"], spacings=spacings,
                             curriculum=tr["curriculum"])
    # Fixed validation crops, the same number for each input, scored per input.
    # They can be smaller than training crops: the validation slab is thinner
    # than the training slab, and the network takes any size.
    val_batches = {}
    for i, name in enumerate(names):
        source = CropSampler([inputs[i]], [labels.data], regions["val"], crop=tr["val_crop"],
                             batch_size=tr["batch_size"], steps_per_epoch=tr["val_batches"],
                             rotate=False, seed=cfg["seed"] + 1,
                             spacings=[spacings[i]] if conditioned else None)
        val_batches[_label(name)] = [(_to_numpy(x), np.asarray(t, np.uint8))
                                     for x, t in (source[j] for j in range(len(source)))]
        del source

    # Model and optimizer: restored, started from a trained model, or fresh.
    loss = build_loss(tr["loss"], class_weights)
    if resuming:
        model = keras.models.load_model(out / "last.keras")
        optimizer = model.optimizer
    else:
        model = unet3d(**cfg["model"], in_channels=n_channels(cfg["data"]["smoothing"]))
        if tr["init_from"]:
            _init_from(model, tr["init_from"], cfg["output"])
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
                 min_delta=tr["early_stopping"]["min_delta"],
                 ema=ema, tb_local=tb_local, state=state, rows=read_log(out / "log.csv")),
    ]
    if first_epoch < tr["epochs"] and state.get("wait", 0) < patience:
        model.fit(train_data, epochs=tr["epochs"], initial_epoch=first_epoch,
                  callbacks=callbacks, verbose=2)
    else:
        print("training already finished")

    evaluate_run(out, catalog)
    return out


def evaluate_run(out: str | Path, catalog: Catalog | None = None, **overrides) -> dict:
    """Predict each input's full box with ``best.keras`` and score the test slab.

    Scores the inputs in ``evaluate.inputs`` (default: the training inputs).
    An input the model was not trained on gets its own normalization, fit
    on the training slab like the others. A model conditioned on spacing is
    given each input's spacing from the catalog. If ``evaluate.labels`` names
    a mask on a finer grid than the training data, predictions are
    upsampled to it and scored on the same test slab at that resolution.

    Writes ``test_scores.json`` (scores and confusion matrix per input) and
    ``test_figure.png`` (a slice through the test slab and the confusion
    matrix for each input), and returns the scores. Keyword arguments override the run's
    ``evaluate`` settings, e.g. ``evaluate_run(out, margin=32)`` to re-score
    a finished run; the results then go to ``test_scores_margin32.json`` and
    ``test_figure_margin32.png`` so the originals are kept.
    """
    out = Path(out)
    cfg = load_config(out / "config.yaml")
    unknown = set(overrides) - set(cfg["evaluate"])
    if unknown:
        raise KeyError(f"unknown evaluate settings {sorted(unknown)}")
    cfg["evaluate"].update(overrides)
    names = cfg["evaluate"]["inputs"] or cfg["data"]["inputs"]
    catalog = catalog or Catalog()
    model = load_model(out / "best.keras")
    labels, mask, make_x, factor = _test_setup(out, cfg, catalog)
    # Slice for the figure: the middle of the test slab, on the scoring grid.
    test = Region(**json.loads((out / "split.json").read_text())["test"])
    mid = (test.start + test.size // 2) * factor
    axis = test.axis

    results, rows = {}, []
    for name in names:
        x = make_x(name)
        spacing = _spacing(catalog, name) if is_conditioned(model) else None
        results[name], pred = _score(model, x, labels, mask, cfg["evaluate"], spacing, factor,
                                     keep=(axis, mid))
        rows.append(dict(name=name, result=results[name], pred=pred,
                         truth=np.take(labels, mid, axis=axis),
                         input=np.take(x[..., -1], mid // factor, axis=axis)))
        _print_scores(name, results[name])
    suffix = "".join(f"_{k}{v}" for k, v in sorted(overrides.items()) if not isinstance(v, (list, dict)))
    (out / f"test_scores{suffix}.json").write_text(json.dumps(results, indent=2))
    _figure(out, rows, suffix, catalog.info(names[0]).get("box_size"))
    return results


def spacing_sweep(out: str | Path, name: str, spacings: list[float],
                  catalog: Catalog | None = None) -> dict:
    """Score one input while telling a conditioned model a range of spacings.

    A check that the model uses its spacing input: if the scores barely
    change when it is told the wrong spacing, the conditioning is doing
    nothing. Writes ``spacing_sweep_<label>.json`` and returns
    ``{spacing: scores}``.
    """
    out = Path(out)
    cfg = load_config(out / "config.yaml")
    catalog = catalog or Catalog()
    model = load_model(out / "best.keras")
    if not is_conditioned(model):
        raise ValueError(f"{out.name} is not conditioned on spacing")
    labels, mask, make_x, factor = _test_setup(out, cfg, catalog)
    x = make_x(name)
    results = {}
    for s in spacings:
        results[str(s)], _ = _score(model, x, labels, mask, cfg["evaluate"], s, factor)
        _print_scores(f"{name} told {s}", results[str(s)])
    (out / f"spacing_sweep_{_label(name)}.json").write_text(json.dumps(results, indent=2))
    return results


def _test_setup(out, cfg, catalog):
    """Scoring labels and test mask, an input builder, and the upsampling factor.

    The builder uses the run's saved normalization; an input the run never
    saw is normalized from its own training slab.
    """
    norms = json.loads((out / "normalization.json").read_text())
    regions = {k: Region(**v) for k, v in json.loads((out / "split.json").read_text()).items()}
    labels = catalog.load(cfg["data"]["labels"]).data
    n = labels.shape[0]
    train_mask = regions["train"].mask(n)

    def make_x(name):
        return build_input(catalog, name, cfg["data"]["smoothing"], norms, train_mask)

    factor = 1
    if cfg["evaluate"]["labels"]:
        labels = catalog.load(cfg["evaluate"]["labels"]).data
        factor, rest = divmod(labels.shape[0], n)
        if rest or factor < 1:
            raise ValueError(f"evaluate.labels grid ({labels.shape[0]}) is not a multiple "
                             f"of the training grid ({n})")
    test = regions["test"]
    test = Region(test.name, test.axis, test.start * factor, test.stop * factor)
    return labels, test.mask(labels.shape[0]), make_x, factor


def _figure(out, rows, suffix, box_size):
    """Test-slab figure for evaluate_run; a failure here never loses the scores."""
    try:
        from .figures import evaluation_figure, save_and_display
        fig = evaluation_figure(rows, f"{out.name}: test slab, middle slice", box_size)
        save_and_display(fig, out / f"test_figure{suffix}.png")
    except Exception as err:                      # noqa: BLE001
        print(f"could not draw the test figure: {err}")


def _score(model, x, labels, mask, ev, spacing=None, factor=1, keep=None):
    """Scores on the test mask, and the predicted slice ``keep = (axis, index)`` if given."""
    if spacing is None:
        fn = model.predict_on_batch
    else:
        def fn(b):
            s = np.full((len(b), 1, 1, 1, 1), spacing, np.float32)
            return model.predict_on_batch({"density": b, "spacing": s})
    probs = predict_volume(fn, x, crop=ev["crop"], margin=ev["margin"],
                           batch_size=ev["batch_size"], tta=ev["tta"])
    pred = probs.argmax(-1) if factor == 1 else upsampled_argmax(probs, factor)
    cm = confusion_matrix(labels, pred, mask=mask)
    kept = np.take(pred, keep[1], axis=keep[0]) if keep else None
    return {**scores(cm), "confusion_matrix": cm.tolist()}, kept


def _print_scores(name, result):
    print(name, {k: round(v, 3) for k, v in result.items()
                 if k in ("mcc", "void_f1", "halo_f1", "balanced_accuracy")})


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
    """One (n, n, n, channels) array per input; normalization saved to the run folder."""
    path = out / "normalization.json"
    norms = json.loads(path.read_text()) if resuming and path.exists() else {}
    arrays = [build_input(catalog, name, cfg["data"]["smoothing"], norms, train_mask)
              for name in cfg["data"]["inputs"]]
    path.write_text(json.dumps(norms, indent=2))
    return arrays


def _spacing(catalog, name):
    spacing = catalog.info(name).get("spacing")
    if spacing is None:
        raise ValueError(f"{name} has no 'spacing' in the catalog, which conditioning needs")
    return float(spacing)


def _label(name):
    """Short label for a dataset name, e.g. 'tng300-3/subhalos/L10' -> 'L10'."""
    return name.rsplit("/", 1)[-1]


def _to_numpy(x):
    if isinstance(x, dict):
        return {k: np.asarray(v) for k, v in x.items()}
    return np.asarray(x)


def _init_from(model, source, output):
    """Copy weights from a saved model (a .keras file or a run folder's best.keras)."""
    path = Path(source)
    if not path.is_absolute():
        path = Path(output) / path
    if path.is_dir():
        path = path / "best.keras"
    copied, skipped = copy_weights(load_model(path), model)
    print(f"initialized {len(copied)} layers from {path}"
          + (f"; left {len(skipped)} as built: {', '.join(skipped)}" if skipped else ""))
    if not copied:
        raise ValueError(f"no layers of {path} match this model")


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
