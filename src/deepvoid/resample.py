"""Moving volumes between grid resolutions.

Used to train on a coarser grid (bigger crops in physical size for the same
cost) while still scoring against the full-resolution mask, so results stay
comparable with full-resolution runs.
"""

from __future__ import annotations

import numpy as np

from .volume import Volume


def block_mean(data: np.ndarray, factor: int) -> np.ndarray:
    """Average over non-overlapping factor^3 blocks: (n, n, n) -> (n/f, n/f, n/f)."""
    n = data.shape[0]
    if n % factor:
        raise ValueError(f"grid size {n} is not divisible by {factor}")
    m = n // factor
    return data.reshape(m, factor, m, factor, m, factor).mean(axis=(1, 3, 5), dtype=np.float64)


def coarsen(volume: Volume, factor: int) -> Volume:
    """A density volume averaged onto a grid ``factor`` times coarser."""
    meta = {**volume.meta, "coarsened_from": volume.n}
    return Volume(data=block_mean(volume.data, factor).astype(np.float32),
                  box_size=volume.box_size, meta=meta)


def upsampled_argmax(probs: np.ndarray, factor: int) -> np.ndarray:
    """Most likely class on a grid ``factor`` times finer.

    Each coarse cell's predicted class fills its factor^3 fine cells. This is
    exactly what the coarse model predicted, with no smoothing. Linear
    interpolation of the probabilities was tried and rejected: it shrinks
    structures one coarse cell across, such as small halo regions, which
    would penalize the coarse model for the upsampling rather than for its
    predictions.
    """
    arg = probs.argmax(-1).astype(np.uint8)
    for axis in range(3):
        arg = np.repeat(arg, factor, axis=axis)
    return arg


def upsample(a: np.ndarray, factor: int) -> np.ndarray:
    """Repeat each cell of a (n, n, n, ...) array into a factor^3 block."""
    for axis in range(3):
        a = np.repeat(a, factor, axis=axis) if factor > 1 else a
    return a
