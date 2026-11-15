"""Training batches drawn straight from whole volumes held on the GPU.

Nothing is pre-cut into patches. Crop origins are drawn so every crop lies
inside the training slab, the crops are gathered in a single GPU call, and
the whole batch gets one of the 48 symmetries of a cube. A 512^3 volume in
float16 is about 0.27 GB, so the data never has to leave the GPU.

With several inputs, each crop in a batch comes from its own randomly chosen
input, with the probabilities set by ``curriculum.py`` (equal by default).
Mixing inputs within a batch matters for batch normalization: if every batch
came from a single input, the batch statistics used in training would
differ from the running averages used at inference, which mix all inputs.
(The first mixed-spacing subhalo run trained well and validated badly for
exactly this reason.) When ``spacings`` are given, each batch is a dict with
the density crops and the tracer spacing of each crop's input, for a model
conditioned on spacing.

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
Z_PERMUTATIONS = tf.constant([0, 5], dtype=tf.int32)   # indices of the two that keep z


class CropSampler(keras.utils.PyDataset):
    """Random crops for ``model.fit``.

    Parameters
    ----------
    inputs, labels:
        One normalized input array, (n, n, n) or (n, n, n, channels), and one
        integer label array, (n, n, n), per volume. Several inputs may share
        a label array.
    region:
        The slab crops are drawn from; they never leave it.
    crop, batch_size, steps_per_epoch:
        Crop side in cells, crops per batch, batches per epoch.
    rotate:
        ``True``: apply one of the 48 symmetries of a cube to each batch.
        ``"z"``: only the 16 that keep the z axis as z (swapping x and y,
        and flipping any axis), for redshift-space inputs whose line of sight
        is z. ``False``: none.
    seed:
        Base seed; with the epoch and batch index it fixes every batch.
    spacings:
        Tracer spacing of each input (h^-1 Mpc). If given, batches are
        ``({"density": x, "spacing": s}, y)`` with s of shape (batch, 1, 1, 1, 1).
    curriculum:
        Sampling weights over epochs, see :func:`input_probabilities`.
    label_copies:
        For a model with several outputs (deep supervision), the labels are
        returned this many times as a tuple, one per output; each output's
        loss coarsens them as needed.
    """

    def __init__(self, inputs: list[np.ndarray], labels: list[np.ndarray], region: Region,
                 crop: int = 96, batch_size: int = 8, steps_per_epoch: int = 200,
                 rotate: bool = True, seed: int = 0, spacings: list[float] | None = None,
                 curriculum: dict | None = None, label_copies: int = 1, **kwargs):
        super().__init__(**kwargs)
        if len(inputs) != len(labels):
            raise ValueError("need one label array per input array")
        if spacings is not None and len(spacings) != len(inputs):
            raise ValueError("need one spacing per input array")
        self.spacings, self.curriculum = spacings, curriculum
        self.label_copies = label_copies
        self.n_inputs = len(inputs)
        self.device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
        # All inputs stacked into one (inputs, n, n, n) tensor so a batch can mix
        # them. Inputs often share one label array, so labels are stacked once
        # per distinct array and each input points at its labels.
        label_ids = list(dict.fromkeys(id(a) for a in labels))
        by_id = {id(a): a for a in labels}
        with tf.device(self.device):
            self.x = tf.identity(tf.constant(np.stack([_with_channels(np.asarray(a, np.float16))
                                                       for a in inputs])))
            self.y = tf.identity(tf.constant(np.stack([np.asarray(by_id[i], np.uint8)
                                                       for i in label_ids])))
        self.label_of = tf.constant([label_ids.index(id(a)) for a in labels], tf.int32)
        self.spacing_of = tf.constant(spacings if spacings is not None else [0.0] * len(inputs),
                                      tf.float32)
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
        v = tf.random.stateless_categorical(self.log_p, self.batch_size, keys[0],
                                            dtype=tf.int32)[0]
        u = tf.random.stateless_uniform([self.batch_size, 3], keys[1])
        origins = self.low + tf.cast(u * tf.cast(self.count, tf.float32), tf.int32)
        x = _gather_crops(self.x, v, origins, self.offsets)
        y = _gather_crops(self.y, tf.gather(self.label_of, v), origins, self.offsets)
        if self.rotate:
            if self.rotate == "z":           # permutations 0 and 5 leave axis 2 in place
                p = tf.gather(Z_PERMUTATIONS,
                              tf.random.stateless_uniform([], keys[2], 0, 2, dtype=tf.int32))
            else:
                p = tf.random.stateless_uniform([], keys[2], 0, 6, dtype=tf.int32)
            flips = tf.random.stateless_uniform([3], keys[3], 0, 2, dtype=tf.int32)
            x, y = _symmetry(x, p, flips), _symmetry(y, p, flips)
        x, y = tf.cast(x, tf.float32), tf.cast(y, tf.int32)
        if self.label_copies > 1:
            y = (y,) * self.label_copies
        if self.spacings is None:
            return x, y
        s = tf.reshape(tf.gather(self.spacing_of, v), [-1, 1, 1, 1, 1])
        return {"density": x, "spacing": s}, y


@tf.function(reduce_retracing=True)
def _gather_crops(volumes, which, origins, offsets):
    """Crop i of side len(offsets) from volumes[which[i]] at origins[i].

    Returns (batch, c, c, c) for label volumes and (batch, c, c, c, channels)
    for input volumes.
    """
    b, c = tf.shape(origins)[0], tf.shape(offsets)[0]
    shape = [b, c, c, c]
    ix = origins[:, 0, None] + offsets
    iy = origins[:, 1, None] + offsets
    iz = origins[:, 2, None] + offsets
    idx = tf.stack([
        tf.broadcast_to(which[:, None, None, None], shape),
        tf.broadcast_to(ix[:, :, None, None], shape),
        tf.broadcast_to(iy[:, None, :, None], shape),
        tf.broadcast_to(iz[:, None, None, :], shape),
    ], axis=-1)
    return tf.gather_nd(volumes, idx)


@tf.function(reduce_retracing=True)
def _symmetry(batch, p, flips):
    """Apply permutation ``PERMUTATIONS[p]`` and the given axis flips to (b, x, y, z, ...)."""
    perm = tf.concat([[0], PERMUTATIONS[p] + 1, tf.range(4, tf.rank(batch))], axis=0)
    batch = tf.transpose(batch, perm)
    axes = tf.boolean_mask(tf.constant([1, 2, 3]), tf.cast(flips, tf.bool))
    return tf.reverse(batch, axes)


def _with_channels(a: np.ndarray) -> np.ndarray:
    return a[..., None] if a.ndim == 3 else a
