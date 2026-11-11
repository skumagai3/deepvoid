"""Full-box prediction from overlapping crops.

The box is covered by crops of side ``crop`` whose centers advance by
``crop - 2 * margin``. Only the central part of each prediction, away from
the crop edges where the network sees little context, is kept. The box is
periodic, so crops wrap around its edges and every voxel is predicted from
a crop in which it sits at least ``margin`` cells from the edge.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def predict_volume(predict_fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray,
                   n_classes: int = 4, crop: int = 128, margin: int = 16,
                   batch_size: int = 4) -> np.ndarray:
    """Class probabilities for every voxel of a periodic box.

    Parameters
    ----------
    predict_fn:
        Maps a batch of shape (B, crop, crop, crop, 1) to probabilities of
        shape (B, crop, crop, crop, n_classes); for a Keras model use
        ``lambda b: model.predict_on_batch(b)``.
    x:
        Normalized input, shape (n, n, n).

    Returns float32 probabilities of shape (n, n, n, n_classes).
    """
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
                          for i, j, k in batch_tiles])[..., None]
        probs = np.asarray(predict_fn(batch.astype(np.float32)))
        keep = probs[:, margin:crop - margin, margin:crop - margin, margin:crop - margin]
        for (i, j, k), p in zip(batch_tiles, keep):
            ii, jj, kk = ((np.arange(o, o + stride) % n) for o in (i, j, k))
            out[np.ix_(ii, jj, kk)] = p
    return out
