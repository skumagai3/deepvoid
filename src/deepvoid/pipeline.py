"""Training batches drawn straight from whole volumes held on the GPU.

Nothing is pre-cut into patches. Each step picks a random volume, a random
crop that lies entirely inside the training region, and a random rotation,
all on the GPU. A 512^3 volume in float16 is about 0.27 GB, so several fit
comfortably next to the model on an L4 or A100, and the data loader never
waits on the CPU or on Google Drive.

Requires TensorFlow.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import keras

from .split import Region


class CropSampler(keras.utils.PyDataset):
    """Endless random crops for ``model.fit``.

    Parameters
    ----------
    inputs, labels:
        Lists of normalized input arrays and integer label arrays, one pair
        per volume, each (n, n, n). Several inputs may share a label array
        (for example the same box at different tracer spacings).
    region:
        The training slab; crops never leave it.
    crop:
        Crop side in cells (64 or 96 is a good size for Colab GPUs).
    """

    def __init__(self, inputs: list[np.ndarray], labels: list[np.ndarray], region: Region,
                 crop: int = 64, batch_size: int = 8, steps_per_epoch: int = 200,
                 rotate: bool = True, seed: int | None = None, **kwargs):
        super().__init__(**kwargs)
        if len(inputs) != len(labels):
            raise ValueError("need one label array per input array")
        device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
        with tf.device(device):  # keep the volumes resident on the GPU
            self.x = [tf.identity(tf.constant(a, dtype=tf.float16)) for a in inputs]
            self.y = [tf.identity(tf.constant(a, dtype=tf.uint8)) for a in labels]
        self.device = device
        n = inputs[0].shape[0]
        self.origins = [tf.constant([r.start, r.stop - r.start], tf.int32)
                        for r in region.crop_origins(crop, n)]
        self.crop, self.batch_size, self.steps = crop, batch_size, steps_per_epoch
        self.rotate = rotate
        self.rng = tf.random.Generator.from_seed(seed) if seed is not None \
            else tf.random.Generator.from_non_deterministic_state()

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, index):
        with tf.device(self.device):
            return self._batch()

    def _batch(self):
        xs, ys = [], []
        for _ in range(self.batch_size):
            v = int(self.rng.uniform([], 0, len(self.x), dtype=tf.int32))
            start = [o[0] + self.rng.uniform([], 0, o[1], dtype=tf.int32) for o in self.origins]
            size = [self.crop] * 3
            x = tf.slice(self.x[v], start, size)
            y = tf.slice(self.y[v], start, size)
            if self.rotate:
                x, y = self._random_rotation(x, y)
            xs.append(x)
            ys.append(y)
        x = tf.cast(tf.stack(xs)[..., None], tf.float32)
        y = tf.cast(tf.stack(ys), tf.int32)
        return x, y

    def _random_rotation(self, x, y):
        """One of the 48 symmetries of a cube (rotations and reflections).

        T-web labels are unchanged by any orthogonal transform of the field,
        so reflections are as valid as rotations for augmentation.
        """
        perms = tf.constant([[0, 1, 2], [1, 2, 0], [2, 0, 1],
                             [0, 2, 1], [2, 1, 0], [1, 0, 2]])
        p = perms[self.rng.uniform([], 0, 6, dtype=tf.int32)]
        flips = self.rng.uniform([3], 0, 2, dtype=tf.int32)
        x, y = tf.transpose(x, p), tf.transpose(y, p)
        for axis in range(3):
            if flips[axis]:
                x, y = tf.reverse(x, [axis]), tf.reverse(y, [axis])
        return x, y
