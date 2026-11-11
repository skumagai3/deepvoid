"""How often each input volume is sampled, epoch by epoch.

With several inputs (for example subhalo fields at spacings 3, 5, 7 and
10 h^-1 Mpc), each training batch comes from one input chosen at random.
By default every input is equally likely. A curriculum changes those
probabilities over training:

    curriculum:
      start: [4, 3, 2, 1]   # relative weights, one per input, at epoch 0
      end: [1, 1, 1, 1]     # weights from epoch `epochs` on
      epochs: 30            # linear change in between

Every input keeps a non-zero weight throughout, so the model never stops
seeing any of them; only the mix changes. This is the "mixed" form of
curriculum, which avoids the forgetting that comes from training on one
input at a time.
"""

from __future__ import annotations

import numpy as np


def input_probabilities(curriculum: dict | None, n_inputs: int, epoch: int) -> np.ndarray:
    """Probability of drawing each input at ``epoch``."""
    if curriculum is None:
        return np.full(n_inputs, 1.0 / n_inputs)
    start = np.asarray(curriculum["start"], float)
    end = np.asarray(curriculum.get("end", [1.0] * n_inputs), float)
    if len(start) != n_inputs or len(end) != n_inputs:
        raise ValueError(f"curriculum weights need one value per input ({n_inputs})")
    if (start <= 0).any() or (end <= 0).any():
        raise ValueError("curriculum weights must be positive so every input is still seen")
    t = min(epoch / max(curriculum.get("epochs", 1), 1), 1.0)
    w = (1 - t) * start / start.sum() + t * end / end.sum()
    return w / w.sum()
