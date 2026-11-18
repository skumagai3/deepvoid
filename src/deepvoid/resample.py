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
    interpolation of the class scores was tried and rejected: it shrinks
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


def onto_coarse_points(data: np.ndarray, offset: int) -> np.ndarray:
    """Values on a grid twice as coarse, averaged around each coarse point.

    The fine grid has point i at position i * d. A coarse point sits on fine
    point 2j + offset: ``offset`` 0 puts coarse point j at 2j * d (point j
    at the corner of coarse cell j, as ``grid.cic`` does), 1 puts it at
    (2j + 1) * d (point j at the cell's center). Each coarse value is the
    fine values weighted 1/4, 1/2, 1/4 along each axis around that point
    (27 fine points in all), so it is centered exactly where the coarse
    point is. ``block_mean`` would instead center it half a fine cell off.

    Works on (n, n, n) or (n, n, n, channels), periodic, one channel at a
    time to bound memory; returns float32 of shape (n/2, n/2, n/2[, channels]).
    """
    if offset not in (0, 1):
        raise ValueError("offset must be 0 or 1")
    n = data.shape[0]
    if n % 2 or data.shape[1] != n or data.shape[2] != n:
        raise ValueError(f"need a cube with an even side; got {data.shape[:3]}")
    if data.ndim == 4:
        return np.stack([onto_coarse_points(data[..., c], offset)
                         for c in range(data.shape[-1])], axis=-1)
    out = np.asarray(data, dtype=np.float32)
    for axis in range(3):
        # centered [1/4, 1/2, 1/4] at the kept points only: x[i-1], x[i], x[i+1]
        keep = np.arange(offset, n, 2)
        out = (0.5 * np.take(out, keep, axis=axis)
               + 0.25 * np.take(out, (keep - 1) % n, axis=axis)
               + 0.25 * np.take(out, (keep + 1) % n, axis=axis))
    return out.astype(np.float32)
