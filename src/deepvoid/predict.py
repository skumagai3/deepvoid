"""Full-box prediction from overlapping crops.

The box is covered by crops of side ``crop`` whose centers advance by
``crop - 2 * margin``. Only the central part of each prediction, away from
the crop edges where the network sees little context, is kept. The box is
periodic, so crops wrap around its edges and every voxel is predicted from
a crop in which it sits at least ``margin`` cells from the edge.

Optional test-time augmentation (TTA) averages the predictions over cube
symmetries of each crop: ``"flips"`` uses the 8 axis reflections and
``"full"`` all 48 rotations and reflections. It costs 8 or 48 times the
prediction time for a small gain in accuracy.
"""

from __future__ import annotations

from typing import Callable

import itertools

import numpy as np


def symmetries(kind: str = "none") -> list[tuple[tuple[int, int, int], tuple[int, ...]]]:
    """Cube symmetries as (axis permutation, flipped axes) pairs."""
    flips = [tuple(a for a, f in enumerate(bits) if f)
             for bits in itertools.product((0, 1), repeat=3)]
    if kind == "none":
        return [((0, 1, 2), ())]
    if kind == "flips":
        return [((0, 1, 2), f) for f in flips]
    if kind == "full":
        return [(p, f) for p in itertools.permutations(range(3)) for f in flips]
    raise ValueError("tta must be 'none', 'flips' or 'full'")


def with_tta(predict_fn: Callable[[np.ndarray], np.ndarray], kind: str = "flips"):
    """Wrap ``predict_fn`` so it averages predictions over cube symmetries."""
    syms = symmetries(kind)

    def predict(batch: np.ndarray) -> np.ndarray:
        total = 0.0
        for perm, flips in syms:
            spatial = [a + 1 for a in flips]
            x = np.flip(np.transpose(batch, (0, *(a + 1 for a in perm), 4)), spatial)
            p = np.flip(np.asarray(predict_fn(np.ascontiguousarray(x))), spatial)
            total = total + np.transpose(p, (0, *(int(i) + 1 for i in np.argsort(perm)), 4))
        return total / len(syms)

    return predict


def predict_volume(predict_fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray,
                   n_classes: int = 4, crop: int = 128, margin: int = 16,
                   batch_size: int = 4, tta: str = "none") -> np.ndarray:
    """Class scores (softmax outputs) for every voxel of a periodic box.

    Parameters
    ----------
    predict_fn:
        Maps a batch of shape (B, crop, crop, crop, channels) to class scores of
        shape (B, crop, crop, crop, n_classes); for a Keras model use
        ``lambda b: model.predict_on_batch(b)``.
    x:
        Normalized input, shape (n, n, n) or (n, n, n, channels).
    tta:
        ``"none"``, ``"flips"`` or ``"full"``; see the module docstring.

    Returns float32 class scores of shape (n, n, n, n_classes), summing to 1
    over the last axis.
    """
    if tta != "none":
        predict_fn = with_tta(predict_fn, tta)
    if x.ndim == 3:
        x = x[..., None]
    n = x.shape[0]
    stride = crop - 2 * margin
    if stride <= 0:
        raise ValueError("crop must be larger than 2 * margin")
    starts = range(0, n, stride)
    tiles = [(i, j, k) for i in starts for j in starts for k in starts]
    out = np.zeros((n, n, n, n_classes), dtype=np.float32)

    def window(o):  # input indices for a tile whose kept region starts at o
        return np.arange(o - margin, o - margin + crop) % n

    for b in range(0, len(tiles), batch_size):
        batch_tiles = tiles[b:b + batch_size]
        batch = np.stack([x[np.ix_(window(i), window(j), window(k))]
                          for i, j, k in batch_tiles])
        probs = np.asarray(predict_fn(batch.astype(np.float32)))
        keep = probs[:, margin:crop - margin, margin:crop - margin, margin:crop - margin]
        for (i, j, k), p in zip(batch_tiles, keep):
            ii, jj, kk = ((np.arange(o, o + stride) % n) for o in (i, j, k))
            out[np.ix_(ii, jj, kk)] = p
    return out
