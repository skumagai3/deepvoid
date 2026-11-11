"""Standard figures.

:func:`evaluation_figure` is drawn by ``train.evaluate_run`` for every
scored run: for each input, a slice through the middle of the test slab
(input, truth, prediction, wrong voxels), the confusion matrix as fractions
of each true class, void precision, recall and MCC on the test slab against
the void-score threshold, and the reliability of the void score. The
prediction and confusion matrix use the void threshold when there is one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .labels import CLASS_NAMES

MASK_COLORS = ["#2b6cb0", "#9ae6b4", "#f6ad55", "#c53030"]


def evaluation_figure(rows: list[dict], title: str, box_size: float | None = None):
    """One row per scored input.

    Each row dict has ``name``, ``input`` (2D), ``truth`` (2D), ``pred``
    (2D) and ``result`` (the scores dict with a ``confusion_matrix``).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(MASK_COLORS)
    curves = any("void_curve" in row["result"] for row in rows)
    rels = any("void_reliability" in row["result"] for row in rows)
    ncol = 5 + curves + rels
    fig, axes = plt.subplots(len(rows), ncol, figsize=(4.4 * ncol, 4.4 * len(rows)), squeeze=False)
    for r, row in enumerate(rows):
        res = row["result"].get("tuned", row["result"])
        threshold = res.get("void_threshold")
        n = row["truth"].shape[0]
        ext = [0, box_size, 0, box_size] if box_size else [0, n, 0, n]
        label = row["name"].rsplit("/", 1)[-1]
        img = row["input"]
        lo, hi = np.percentile(img, [1, 99.7])
        if hi <= lo:                                  # a sparse raw field is mostly one value
            hi = img.max() if img.max() > lo else lo + 1
        axes[r, 0].imshow(img.T, origin="lower", extent=ext, cmap="magma", vmin=lo, vmax=hi)
        axes[r, 0].set_title(f"{label}: input (last channel)")
        pred_title = "prediction" if threshold is None else f"prediction (void if score ≥ {threshold:.2f})"
        for c, (key, name) in enumerate((("truth", "truth"), ("pred", pred_title)), start=1):
            axes[r, c].imshow(row[key].T, origin="lower", extent=ext, cmap=cmap,
                              vmin=-0.5, vmax=3.5, interpolation="nearest")
            axes[r, c].set_title(name)
        wrong = np.where(row["pred"] != row["truth"], row["truth"], np.nan)
        axes[r, 3].imshow(wrong.T, origin="lower", extent=ext, cmap=cmap, vmin=-0.5, vmax=3.5,
                          interpolation="nearest")
        axes[r, 3].set_facecolor("#eeeeee")
        axes[r, 3].set_title("wrong voxels (true class)")
        f1 = "  ".join(f"{k[0]} {res[f'{k}_f1']:.2f}" for k in CLASS_NAMES)
        axes[r, 2].set_xlabel(f"void MCC {res['void_mcc']:.3f}   MCC {res['mcc']:.3f}\n"
                              f"F1: {f1}", fontsize=10)

        cm = np.asarray(res["confusion_matrix"], float)
        frac = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax = axes[r, 4]
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                ax.text(j, i, f"{frac[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if frac[i, j] > 0.5 else "black")
        ax.set_xticks(range(4), CLASS_NAMES, fontsize=8)
        ax.set_yticks(range(4), CLASS_NAMES, fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title("confusion (fraction of true class)")

        curve = row["result"].get("void_curve", {}).get("test")
        if curves and curve:
            ax = axes[r, 5]
            t = np.asarray(curve["thresholds"])
            for key, color in (("precision", "#2b6cb0"), ("recall", "#dd6b20"), ("mcc", "k")):
                ax.plot(t, curve[key], color=color, label=f"void {key}")
            if threshold is not None:
                ax.axvline(threshold, color="grey", ls="--", lw=1, label=f"chosen {threshold:.2f}")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
            ax.set_xlabel("threshold on void score")
            ax.set_title("void vs rest on the test slab")
            ax.legend(fontsize=8, loc="lower left")

        rel = row["result"].get("void_reliability")
        if rels and rel:
            ax = axes[r, 5 + curves]
            mean = np.array(rel["mean_score"], dtype=float)
            frac = np.array(rel["fraction_void"], dtype=float)
            ax.plot([0, 1], [0, 1], "k:", lw=1)
            ax.plot(mean, frac, "-o", ms=4, color="#2b6cb0")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
            ax.set_xlabel("mean void score in bin")
            ax.set_ylabel("fraction labelled void")
            ax.set_title(f"void score reliability (ECE {rel['ece']:.3f})")
    for ax in axes[:, :4].ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return fig


def save_and_display(fig, path: str | Path) -> Path:
    """Save a figure; show it inline when running in a notebook."""
    import matplotlib.pyplot as plt
    path = Path(path)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    try:
        from IPython import get_ipython
        from IPython.display import Image, display
        if get_ipython() is not None:
            display(Image(filename=str(path)))
    except ImportError:
        pass
    return path
