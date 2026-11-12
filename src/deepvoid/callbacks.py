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
from .stopping import Stopper

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

    ``batches`` maps a label (one per input, e.g. ``"L10"``) to that input's
    list of (x, y) batches. Adds ``val_loss`` (unweighted cross-entropy),
    ``val_accuracy``, ``val_mcc`` (four classes), ``val_void_mcc`` (void
    against the rest), ``val_void_f1`` and ``val_halo_f1``, pooled over all
    inputs, to the epoch logs, and with more than one input also
    ``val_mcc_<label>`` and ``val_void_mcc_<label>`` for each. Writes a
    validation slice (truth above, prediction below) to TensorBoard. Uses the averaged weights when
    an :class:`EMA` is given.
    """

    def __init__(self, batches: dict[str, list], log_dir: Path, ema: EMA | None = None,
                 predictor=None):
        super().__init__()
        self.batches, self.ema, self.predictor = batches, ema, predictor
        self.writer = tf.summary.create_file_writer(str(log_dir / "validation"))

    def on_epoch_end(self, epoch, logs=None):
        total, nll, count, first, per_input = 0, 0.0, 0, None, {}
        with self.ema.swapped() if self.ema else contextlib.nullcontext():
            for label, batches in self.batches.items():
                cm = 0
                for x, y in batches:
                    y = np.asarray(y)
                    p = np.asarray((self.predictor or self.model).predict_on_batch(x),
                                   dtype=np.float32)
                    pred = p.argmax(-1)
                    cm = cm + confusion_matrix(y, pred)
                    nll += -np.log(np.clip(np.take_along_axis(p, y[..., None], -1),
                                           1e-7, 1)).sum()
                    count += y.size
                    if first is None:
                        first = (y[0], pred[0])
                per_input[label] = scores(cm)
                total = total + cm
        s = scores(total)
        if logs is not None:
            logs.update(val_loss=nll / count, val_accuracy=s["accuracy"], val_mcc=s["mcc"],
                        val_void_mcc=s["void_mcc"], val_void_f1=s["void_f1"],
                        val_halo_f1=s["halo_f1"])
            if len(per_input) > 1:
                logs.update({f"val_mcc_{k}": v["mcc"] for k, v in per_input.items()})
                logs.update({f"val_void_mcc_{k}": v["void_mcc"] for k, v in per_input.items()})
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
    - stops training once the score has not improved by at least
      ``min_delta`` for ``patience`` epochs (see ``stopping.py``).
    """

    def __init__(self, out: Path, sampler, monitor: str = "val_mcc", patience: int = 15,
                 min_delta: float = 0.0, ema: EMA | None = None,
                 tb_local: Path | None = None, state: dict | None = None,
                 rows: list[dict] | None = None, best_model=None):
        super().__init__()
        self.out, self.sampler, self.ema = out, sampler, ema
        self.best_model = best_model
        self.monitor, self.tb_local = monitor, tb_local
        self.stopper = Stopper("min" if "loss" in monitor else "max", patience, min_delta, state)
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
        if self.stopper.update(epoch, value):
            with self.ema.swapped() if self.ema else contextlib.nullcontext():
                _atomic_save(self.best_model or self.model, self.out / "best.keras")

        _atomic_save(self.model, self.out / "last.keras")
        if self.ema:
            self.ema.save(self.out / "ema.npz")
        state = {"epoch": epoch, **self.stopper.state(), "monitor": self.monitor}
        (self.out / "state.json").write_text(json.dumps(state, indent=2))
        if self.tb_local is not None:
            shutil.copytree(self.tb_local, self.out / "tensorboard", dirs_exist_ok=True)

        st = self.stopper
        print(f"epoch {epoch}: {self.monitor} {value:.4f} "
              f"(best {st.best:.4f} at epoch {st.best_epoch}; "
              f"{st.wait}/{st.patience} epochs without a gain of {st.min_delta})")
        if st.should_stop:
            print(f"stopping: {self.monitor} has not improved by {st.min_delta} "
                  f"in {st.patience} epochs")
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
