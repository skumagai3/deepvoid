"""Voxel-level scores.

Scores are computed from a confusion matrix, optionally restricted to a
region (the held-out slab) so that training voxels never count.
Catalog-level validation (void sizes, centers, profiles) will live here too.
"""

from __future__ import annotations

import numpy as np

from .labels import CLASS_NAMES


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4,
                     mask: np.ndarray | None = None) -> np.ndarray:
    """Counts with true classes as rows and predicted classes as columns."""
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    idx = y_true.astype(np.int64).ravel() * n_classes + y_pred.astype(np.int64).ravel()
    return np.bincount(idx, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def scores(cm: np.ndarray, class_names=CLASS_NAMES) -> dict:
    """Per-class precision, recall and F1, plus accuracy, balanced accuracy and MCC.

    MCC is the multiclass generalization of Gorodkin (2004).
    """
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    true_counts, pred_counts, total = cm.sum(1), cm.sum(0), cm.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(pred_counts > 0, tp / pred_counts, 0.0)
        recall = np.where(true_counts > 0, tp / true_counts, 0.0)
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / (precision + recall), 0.0)
    cov_tp = tp.sum() * total - pred_counts @ true_counts
    denom = np.sqrt((total ** 2 - pred_counts @ pred_counts) *
                    (total ** 2 - true_counts @ true_counts))
    present = true_counts > 0
    out = {
        "accuracy": tp.sum() / total,
        "balanced_accuracy": recall[present].mean(),
        "mcc": cov_tp / denom if denom > 0 else 0.0,
    }
    for name, p, r, f in zip(class_names, precision, recall, f1):
        out[f"{name}_precision"], out[f"{name}_recall"], out[f"{name}_f1"] = p, r, f
    return {k: float(v) for k, v in out.items()}
