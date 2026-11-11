"""Voxel-level scores.

Scores are computed from a confusion matrix, optionally restricted to a
region (the held-out slab) so that training voxels never count.

Voids are the science product, so besides the four-class scores every
result has two-class void scores (void against everything else):
``void_mcc`` and ``void_iou``.

The network's output for a voxel is the softmax of its last layer: four
non-negative numbers that sum to 1, here called class scores (the first is
the void score). They are not probabilities of anything by construction.
The void call can be made with a threshold on the void score instead of
taking the highest score; :func:`void_threshold_curve` and
:func:`best_void_threshold` choose the threshold on the validation slab.
That needs only that a higher void score means "more void-like", not that
the score is a probability. Whether a score can be read as a relative
frequency is a separate, empirical question: see :func:`void_reliability`.

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
    void = binary_scores(tp=cm[0, 0], fp=cm[1:, 0].sum(), fn=cm[0, 1:].sum(), tn=cm[1:, 1:].sum())
    out["void_mcc"], out["void_iou"] = void["mcc"], void["iou"]
    return {k: float(v) for k, v in out.items()}


def binary_scores(tp, fp, fn, tn) -> dict:
    """Precision, recall, F1, IoU and MCC for one class against the rest (arrays allowed)."""
    tp, fp, fn, tn = (np.asarray(v, np.float64) for v in (tp, fp, fn, tn))
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), 0.0)
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.where(denom > 0, (tp * tn - fp * fn) / denom, 0.0)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou, "mcc": mcc}


THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


def void_threshold_curve(p_void: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None,
                         thresholds: np.ndarray = THRESHOLDS) -> dict:
    """Two-class void scores when a voxel is called void if its void score >= t, for each t.

    Uses histograms of the void score for true-void and other voxels, so it
    costs one pass over the data however many thresholds are tried.
    """
    p = p_void if mask is None else p_void[mask]
    is_void = (truth if mask is None else truth[mask]) == 0
    edges = np.concatenate([[-np.inf], thresholds, [np.inf]])
    void_hist = np.histogram(p[is_void], edges)[0]
    other_hist = np.histogram(p[~is_void], edges)[0]
    # Voxels with p >= thresholds[i] fall in bins i+1 and above.
    tp = np.cumsum(void_hist[::-1])[::-1][1:]
    fp = np.cumsum(other_hist[::-1])[::-1][1:]
    fn, tn = is_void.sum() - tp, (~is_void).sum() - fp
    return {"thresholds": thresholds, **binary_scores(tp, fp, fn, tn)}


def best_void_threshold(curve: dict, objective: str = "mcc") -> float:
    """Threshold that maximizes ``objective`` (e.g. 'mcc', 'f1', 'iou') on a curve."""
    return float(curve["thresholds"][int(np.argmax(curve[objective]))])


def decide(probs: np.ndarray, void_threshold: float | None = None) -> np.ndarray:
    """Class labels from class scores (..., 4).

    Without a threshold, the class with the highest score. With one, void
    wherever the void score >= threshold, and otherwise the non-void class
    with the highest score.
    """
    if void_threshold is None:
        return probs.argmax(-1).astype(np.uint8)
    rest = probs[..., 1:].argmax(-1).astype(np.uint8) + 1
    return np.where(probs[..., 0] >= void_threshold, 0, rest).astype(np.uint8)


def precision_at_recall(curve: dict, recall: float) -> float:
    """Highest void precision among thresholds whose void recall is at least ``recall``.

    Read off a :func:`void_threshold_curve`; NaN if no threshold reaches
    that recall.
    """
    ok = np.asarray(curve["recall"]) >= recall
    return float(np.max(np.asarray(curve["precision"])[ok])) if ok.any() else float("nan")


def recall_at_precision(curve: dict, precision: float) -> float:
    """Highest void recall among thresholds whose void precision is at least ``precision``."""
    ok = np.asarray(curve["precision"]) >= precision
    return float(np.max(np.asarray(curve["recall"])[ok])) if ok.any() else float("nan")


def void_reliability(s_void: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None,
                     n_bins: int = 10) -> dict:
    """Is the void score a calibrated estimate of the void fraction?

    Voxels are grouped into ``n_bins`` equal-width bins of void score. For
    each bin: the number of voxels, their mean void score, and the fraction
    of them labelled void in the T-web mask. If the two agree in every bin,
    the void score can be read as the frequency with which voxels given that
    score are void, for this set of voxels (one simulation, one tracer
    sample); it says nothing about other data. The expected calibration
    error is ECE = sum_b (n_b / N) |fraction_void_b - mean_score_b|.
    """
    s = s_void if mask is None else s_void[mask]
    is_void = (truth if mask is None else truth[mask]) == 0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts = np.histogram(s, edges)[0].astype(np.float64)
    score_sum = np.histogram(s, edges, weights=s.astype(np.float64))[0]
    void_count = np.histogram(s[is_void], edges)[0].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_score = np.where(counts > 0, score_sum / counts, np.nan)
        frac_void = np.where(counts > 0, void_count / counts, np.nan)
    filled = counts > 0
    ece = float(np.sum(counts[filled] / counts.sum()
                       * np.abs(frac_void[filled] - mean_score[filled])))
    return {"bin_edges": edges, "counts": counts, "mean_score": mean_score,
            "fraction_void": frac_void, "ece": ece}
