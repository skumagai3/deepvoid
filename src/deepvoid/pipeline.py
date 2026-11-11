"""Training batches drawn straight from whole volumes held on the GPU.

Nothing is pre-cut into patches. Each batch comes from one randomly chosen
volume: crop origins are drawn so every crop lies inside the training slab,
the crops are gathered in a single GPU call, and the whole batch gets one of
the 48 symmetries of a cube. A 512^3 volume in float16 is about 0.27 GB, so
the data never has to leave the GPU.

With several inputs, the input for each batch is drawn with the
probabilities set by ``curriculum.py`` (equal by default). When
``spacings`` are given, each batch is a dict with the density crops and the
tracer spacing of the input they came from, for a model conditioned on
spacing.

Randomness is stateless: batch ``i`` of epoch ``e`` always uses the same
crops for a given seed, which makes resumed runs see the same data they
would have seen without the interruption.

Requires TensorFlow.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import keras

from .curriculum import input_probabilities
from .split import Region

# Axis orders for the 6 permutations of (x, y, z); with the 8 sign flips
# these give the 48 symmetries of a cube.
PERMUTATIONS = tf.constant([[0, 1, 2], [1, 2, 0], [2, 0, 1],
                            [0, 2, 1], [2, 1, 0], [1, 0, 2]], dtype=tf.int32)


class CropSampler(keras.utils.PyDataset):
    """Random crops for ``model.fit``.

    Parameters
    ----------
    inputs, labels:
        One normalized input array and one integer label array per volume,
        each (n, n, n). Several inputs may share a label array.
    region:
        The slab crops are drawn from; they never leave it.
    crop, batch_size, steps_per_epoch:
        Crop side in cells, crops per batch, batches per epoch.
    rotate:
        Apply a random cube symmetry to each batch.
    seed:
        Base seed; with the epoch and batch index it fixes every batch.
    spacings:
        Tracer spacing of each input (h^-1 Mpc). If given, batches are
        ``({"density": x, "spacing": s}, y)`` with s of shape (batch, 1, 1, 1, 1).
    curriculum:
        Sampling weights over epochs, see :func:`input_probabilities`.
    """

    def __init__(self, inputs: list[np.ndarray], labels: list[np.ndarray], region: Region,
                 crop: int = 96, batch_size: int = 8, steps_per_epoch: int = 200,
                 rotate: bool = True, seed: int = 0, spacings: list[float] | None = None,
                 curriculum: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        if len(inputs) != len(labels):
            raise ValueError("need one label array per input array")
        if spacings is not None and len(spacings) != len(inputs):
            raise ValueError("need one spacing per input array")
        self.spacings, self.curriculum = spacings, curriculum
        self.n_inputs = len(inputs)
        self.device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
        with tf.device(self.device):
            self.x = [tf.identity(tf.constant(a, dtype=tf.float16)) for a in inputs]
            self.y = [tf.identity(tf.constant(a, dtype=tf.uint8)) for a in labels]
        n = inputs[0].shape[0]
        ranges = region.crop_origins(crop, n)
        self.low = tf.constant([r.start for r in ranges], tf.int32)
        self.count = tf.constant([len(r) for r in ranges], tf.int32)
        self.crop, self.batch_size, self.steps = crop, batch_size, steps_per_epoch
        self.rotate, self.seed = rotate, seed
        self.offsets = tf.range(crop, dtype=tf.int32)
        self.set_epoch(0)

    def __len__(self) -> int:
        return self.steps

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        p = input_probabilities(self.curriculum, self.n_inputs, epoch)
        self.log_p = tf.constant(np.log(p)[None], tf.float32)

    def __getitem__(self, index):
        key = tf.constant([self.seed, self.epoch * self.steps + index], tf.int64)
        with tf.device(self.device):
            return self.batch(key)

    def batch(self, key):
        """One batch for a stateless random key of shape (2,)."""
        keys = tf.random.experimental.stateless_split(key, num=4)
        v = int(tf.random.stateless_categorical(self.log_p, 1, keys[0])[0, 0])
        u = tf.random.stateless_uniform([self.batch_size, 3], keys[1])
        origins = self.low + tf.cast(u * tf.cast(self.count, tf.float32), tf.int32)
        x = _gather_crops(self.x[v], origins, self.offsets)
        y = _gather_crops(self.y[v], origins, self.offsets)
        if self.rotate:
            p = tf.random.stateless_uniform([], keys[2], 0, 6, dtype=tf.int32)
            flips = tf.random.stateless_uniform([3], keys[3], 0, 2, dtype=tf.int32)
            x, y = _symmetry(x, p, flips), _symmetry(y, p, flips)
        x, y = tf.cast(x[..., None], tf.float32), tf.cast(y, tf.int32)
        if self.spacings is None:
            return x, y
        s = tf.fill([self.batch_size, 1, 1, 1, 1], float(self.spacings[v]))
        return {"density": x, "spacing": s}, y


@tf.function(reduce_retracing=True)
def _gather_crops(volume, origins, offsets):
    """Crops of side len(offsets) at each origin, shape (batch, c, c, c)."""
    ix = origins[:, 0, None] + offsets
    iy = origins[:, 1, None] + offsets
    iz = origins[:, 2, None] + offsets
    c = tf.shape(offsets)[0]
    idx = tf.stack([
        tf.broadcast_to(ix[:, :, None, None], [tf.shape(origins)[0], c, c, c]),
        tf.broadcast_to(iy[:, None, :, None], [tf.shape(origins)[0], c, c, c]),
        tf.broadcast_to(iz[:, None, None, :], [tf.shape(origins)[0], c, c, c]),
    ], axis=-1)
    return tf.gather_nd(volume, idx)


@tf.function(reduce_retracing=True)
def _symmetry(batch, p, flips):
    """Apply permutation ``PERMUTATIONS[p]`` and the given axis flips to (b, x, y, z)."""
    perm = tf.concat([[0], PERMUTATIONS[p] + 1], axis=0)
    batch = tf.transpose(batch, perm)
    axes = tf.boolean_mask(tf.constant([1, 2, 3]), tf.cast(flips, tf.bool))
    return tf.reverse(batch, axes)
