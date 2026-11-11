"""Train a model from a config file.

    python -m deepvoid.train configs/tng_dm_baseline.yaml

or, in a notebook, ``deepvoid.train.run("configs/tng_dm_baseline.yaml")``.

A run directory gets the config, the fitted normalization, a CSV training
log and the best checkpoint, so every result can be traced back to exactly
what produced it. Requires TensorFlow.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import keras
import numpy as np
import yaml

from .catalog import Catalog
from .losses import WeightedSCCE
from .models import unet3d
from .pipeline import CropSampler
from .preprocess import Normalizer
from .split import slab_split


def run(config_path: str, catalog: Catalog | None = None) -> Path:
    cfg = yaml.safe_load(open(config_path))
    data, model_cfg, tr = cfg["data"], cfg["model"], cfg["training"]
    catalog = catalog or Catalog()
    keras.utils.set_random_seed(tr.get("seed", 0))
    if tr.get("mixed_precision", True):
        keras.mixed_precision.set_global_policy("mixed_float16")

    out = Path(cfg["output"]) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out / "config.yaml")

    labels = catalog.load(data["labels"])
    regions = slab_split(labels.n, **data.get("split", {}))
    train_mask = regions["train"].mask(labels.n)

    # One normalizer per input field, fit on the training slab only.
    inputs, norms = [], {}
    for name in data["inputs"]:
        vol = catalog.load(name)
        norms[name] = Normalizer.fit(vol.data, train_mask)
        inputs.append(norms[name](vol.data))
    (out / "normalization.json").write_text(
        json.dumps({k: v.to_dict() for k, v in norms.items()}, indent=2))
    (out / "split.json").write_text(json.dumps({k: vars(v) for k, v in regions.items()}))

    y = [labels.data] * len(inputs)
    sampler = dict(crop=tr["crop"], batch_size=tr["batch_size"])
    train_data = CropSampler(inputs, y, regions["train"], steps_per_epoch=tr["steps_per_epoch"],
                             seed=tr.get("seed"), **sampler)
    # A fixed set of validation crops, so val loss is comparable across epochs.
    val_source = CropSampler(inputs, y, regions["val"], steps_per_epoch=tr["val_steps"],
                             rotate=False, seed=1234, **sampler)
    val_batches = [val_source[i] for i in range(len(val_source))]

    model = unet3d(**model_cfg)
    model.compile(optimizer=keras.optimizers.Adam(tr["learning_rate"]),
                  loss=WeightedSCCE(tr.get("class_weights")),
                  metrics=["sparse_categorical_accuracy"])
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(out / "best.keras"), save_best_only=True),
        keras.callbacks.CSVLogger(str(out / "log.csv")),
        keras.callbacks.ReduceLROnPlateau(factor=0.25, patience=10),
        keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True),
    ]
    model.fit(train_data, validation_data=_as_dataset(val_batches),
              epochs=tr["epochs"], callbacks=callbacks)
    return out


def _as_dataset(batches):
    import tensorflow as tf
    x = np.concatenate([b[0] for b in batches])
    y = np.concatenate([b[1] for b in batches])
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(len(batches[0][0]))


if __name__ == "__main__":
    print(run(sys.argv[1]))
