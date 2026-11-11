"""Confidence: making predicted probabilities mean what they say.

Temperature scaling (Guo et al. 2017) fits one number T on held-out voxels
so that softmax(logits / T) is calibrated; a voxel predicted void with
probability 0.8 should then be void about 80% of the time. It changes
confidences, never the predicted class.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale probabilities as softmax(log(p) / T) along the last axis."""
    z = np.log(np.clip(probs, 1e-7, 1.0)) / temperature
    z -= z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit_temperature(probs: np.ndarray, labels: np.ndarray, max_voxels: int = 2_000_000,
                    seed: int = 0) -> float:
    """Temperature minimizing the negative log-likelihood of ``labels``.

    ``probs`` has shape (..., n_classes) and ``labels`` the matching integer
    shape; use held-out voxels only. A random subsample keeps this fast.
    """
    p = probs.reshape(-1, probs.shape[-1])
    y = labels.reshape(-1).astype(np.int64)
    if len(y) > max_voxels:
        pick = np.random.default_rng(seed).choice(len(y), max_voxels, replace=False)
        p, y = p[pick], y[pick]

    def nll(t):
        q = apply_temperature(p, t)
        return -np.mean(np.log(np.clip(q[np.arange(len(y)), y], 1e-12, 1.0)))

    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def reliability(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """Reliability diagram data and expected calibration error (ECE).

    Uses the top-class confidence of each voxel.
    """
    p = probs.reshape(-1, probs.shape[-1])
    conf = p.max(axis=1)
    correct = p.argmax(axis=1) == labels.reshape(-1)
    edges = np.linspace(0, 1, n_bins + 1)
    which = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(which, minlength=n_bins)
    with np.errstate(invalid="ignore"):
        acc = np.bincount(which, weights=correct, minlength=n_bins) / counts
        mean_conf = np.bincount(which, weights=conf, minlength=n_bins) / counts
    filled = counts > 0
    ece = np.sum(counts[filled] / counts.sum() * np.abs(acc[filled] - mean_conf[filled]))
    return {"bin_edges": edges, "accuracy": acc, "confidence": mean_conf,
            "counts": counts, "ece": float(ece)}
