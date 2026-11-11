"""Training callbacks: weight averaging, validation scores, and run state.

Requires TensorFlow / Keras 3.
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import shutil
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from .evaluate import confusion_matrix, scores

PALETTE = np.array([[43, 108, 176], [154, 230, 180], [246, 173, 85], [197, 48, 48]],
                   dtype=np.uint8)  # void, wall, filament, halo


class EMA(keras.callbacks.Callback):
    """Exponential moving average of the model weights.

    After every step, ``shadow = d * shadow + (1 - d) * weights``, where the
    decay d ramps up to ``decay`` over the first steps so early averages are
    not dominated by the random initialization. The averaged weights are
    usually smoother and slightly better than the raw ones; they are used
    for validation and for the saved best model, while training continues on
    the raw weights.
    """

    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.shadow = None
        self.step = tf.Variable(0, dtype=tf.int64, trainable=False)
        self._pending = None

    def on_train_begin(self, logs=None):
        if self.shadow is None:
            # Float weights only: this skips integer states such as RNG seeds.
            self.vars = [w for w in self.model.weights if "float" in str(w.dtype)]
            self.shadow = [tf.Variable(w, trainable=False) for w in self.vars]
            if self._pending is not None:
                self._set(self._pending)
            self._update = tf.function(self._update_fn)

    def _update_fn(self):
        self.step.assign_add(1)
        step = tf.cast(self.step, tf.float32)
        d = tf.minimum(self.decay, (1.0 + step) / (10.0 + step))
        for s, w in zip(self.shadow, self.vars):
            s.assign(d * s + (1.0 - d) * tf.cast(w, s.dtype))

    def on_train_batch_end(self, batch, logs=None):
        self._update()

    @contextlib.contextmanager
    def swapped(self):
        """Temporarily put the averaged weights into the model."""
        raw = [w.numpy() for w in self.vars]
        for s, w in zip(self.shadow, self.vars):
            w.assign(s)
        try:
            yield
        finally:
            for r, w in zip(raw, self.vars):
                w.assign(r)

    def save(self, path: Path) -> None:
        arrays = {f"w{i}": s.numpy() for i, s in enumerate(self.shadow)}
        np.savez(path, step=self.step.numpy(), **arrays)

    def load(self, path: Path) -> None:
        with np.load(path) as f:
            self.step.assign(int(f["step"]))
            self._pending = [f[f"w{i}"] for i in range(len(f.files) - 1)]

    def _set(self, arrays):
        for s, a in zip(self.shadow, arrays):
            s.assign(a)


class Validation(keras.callbacks.Callback):
    """Scores the fixed validation crops at the end of every epoch.

    Adds ``val_loss`` (unweighted cross-entropy), ``val_accuracy``,
    ``val_mcc``, ``val_void_f1`` and ``val_halo_f1`` to the epoch logs, and
    writes a validation slice (truth above, prediction below) to TensorBoard.
    Uses the averaged weights when an :class:`EMA` is given.
    """

    def __init__(self, batches, log_dir: Path, ema: EMA | None = None):
        super().__init__()
        self.batches, self.ema = batches, ema
        self.writer = tf.summary.create_file_writer(str(log_dir / "validation"))

    def on_epoch_end(self, epoch, logs=None):
        with self.ema.swapped() if self.ema else contextlib.nullcontext():
            cm, nll, count, first = 0, 0.0, 0, None
            for x, y in self.batches:
                y = np.asarray(y)
                p = np.asarray(self.model.predict_on_batch(x), dtype=np.float32)
                pred = p.argmax(-1)
                cm = cm + confusion_matrix(y, pred)
                nll += -np.log(np.clip(np.take_along_axis(p, y[..., None], -1), 1e-7, 1)).sum()
                count += y.size
                if first is None:
                    first = (y[0], pred[0])
        s = scores(cm)
        if logs is not None:
            logs.update(val_loss=nll / count, val_accuracy=s["accuracy"], val_mcc=s["mcc"],
                        val_void_f1=s["void_f1"], val_halo_f1=s["halo_f1"])
        k = first[0].shape[0] // 2
        image = np.concatenate([PALETTE[first[0][k]], PALETTE[first[1][k]]], axis=0)
        with self.writer.as_default():
            tf.summary.image("validation slice: truth (top), prediction (bottom)",
                             image[None], step=epoch)
        self.writer.flush()


class RunState(keras.callbacks.Callback):
    """Checkpoints, early stopping, the CSV log and everything needed to resume.

    At the end of each epoch it

    - records the learning rate and writes the full ``log.csv`` again, so the
      copy on Google Drive is always complete;
    - saves ``best.keras`` (with averaged weights if an EMA is used) when the
      monitored score improves;
    - saves ``last.keras`` (raw weights and optimizer state), the EMA weights
      and ``state.json``, which together let a stopped run continue;
    - copies the TensorBoard logs from fast local disk to the run folder;
    - stops training once the score has not improved for ``patience`` epochs.
    """

    def __init__(self, out: Path, sampler, monitor: str = "val_mcc", patience: int = 15,
                 ema: EMA | None = None, tb_local: Path | None = None,
                 state: dict | None = None, rows: list[dict] | None = None):
        super().__init__()
        self.out, self.sampler, self.ema = out, sampler, ema
        self.monitor, self.patience, self.tb_local = monitor, patience, tb_local
        self.mode = "min" if "loss" in monitor else "max"
        state = state or {}
        self.best = state.get("best", np.inf if self.mode == "min" else -np.inf)
        self.best_epoch = state.get("best_epoch", -1)
        self.wait = state.get("wait", 0)
        self.rows = rows or []

    def on_epoch_begin(self, epoch, logs=None):
        self.sampler.set_epoch(epoch)

    def on_epoch_end(self, epoch, logs=None):
        logs = dict(logs or {})
        logs["learning_rate"] = float(keras.ops.convert_to_numpy(
            self.model.optimizer.learning_rate))
        self.rows.append({"epoch": epoch, **{k: float(v) for k, v in logs.items()}})
        self._write_log()

        value = logs[self.monitor]
        improved = value < self.best if self.mode == "min" else value > self.best
        if improved:
            self.best, self.best_epoch, self.wait = float(value), epoch, 0
            with self.ema.swapped() if self.ema else contextlib.nullcontext():
                _atomic_save(self.model, self.out / "best.keras")
        else:
            self.wait += 1

        _atomic_save(self.model, self.out / "last.keras")
        if self.ema:
            self.ema.save(self.out / "ema.npz")
        state = {"epoch": epoch, "best": self.best, "best_epoch": self.best_epoch,
                 "wait": self.wait, "monitor": self.monitor}
        (self.out / "state.json").write_text(json.dumps(state, indent=2))
        if self.tb_local is not None:
            shutil.copytree(self.tb_local, self.out / "tensorboard", dirs_exist_ok=True)

        print(f"epoch {epoch}: {self.monitor} {value:.4f} "
              f"(best {self.best:.4f} at epoch {self.best_epoch})")
        if self.wait >= self.patience:
            print(f"stopping: no improvement in {self.monitor} for {self.patience} epochs")
            self.model.stop_training = True

    def _write_log(self):
        fields = list(dict.fromkeys(k for row in self.rows for k in row))
        tmp = self.out / "log.csv.tmp"
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)
        os.replace(tmp, self.out / "log.csv")


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [{k: float(v) if k != "epoch" else int(float(v)) for k, v in row.items() if v}
                for row in csv.DictReader(f)]


def _atomic_save(model, path: Path) -> None:
    tmp = path.with_name(path.stem + ".tmp.keras")
    model.save(tmp)
    os.replace(tmp, path)
