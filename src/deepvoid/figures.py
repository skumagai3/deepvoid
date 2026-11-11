"""Standard figures.

:func:`evaluation_figure` is drawn by ``train.evaluate_run`` for every
scored run: for each input, a slice through the middle of the test slab
(input, truth, prediction, wrong voxels) and the confusion matrix as
fractions of each true class.
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
    fig, axes = plt.subplots(len(rows), 5, figsize=(22, 4.4 * len(rows)), squeeze=False)
    for r, row in enumerate(rows):
        res = row["result"]
        n = row["truth"].shape[0]
        ext = [0, box_size, 0, box_size] if box_size else [0, n, 0, n]
        label = row["name"].rsplit("/", 1)[-1]
        img = row["input"]
        lo, hi = np.percentile(img, [1, 99.7])
        if hi <= lo:                                  # a sparse raw field is mostly one value
            hi = img.max() if img.max() > lo else lo + 1
        axes[r, 0].imshow(img.T, origin="lower", extent=ext, cmap="magma", vmin=lo, vmax=hi)
        axes[r, 0].set_title(f"{label}: input (last channel)")
        for c, (key, name) in enumerate((("truth", "truth"), ("pred", "prediction")), start=1):
            axes[r, c].imshow(row[key].T, origin="lower", extent=ext, cmap=cmap,
                              vmin=-0.5, vmax=3.5, interpolation="nearest")
            axes[r, c].set_title(name)
        wrong = np.where(row["pred"] != row["truth"], row["truth"], np.nan)
        axes[r, 3].imshow(wrong.T, origin="lower", extent=ext, cmap=cmap, vmin=-0.5, vmax=3.5,
                          interpolation="nearest")
        axes[r, 3].set_facecolor("#eeeeee")
        axes[r, 3].set_title("wrong voxels (true class)")
        f1 = "  ".join(f"{k[0]} {res[f'{k}_f1']:.2f}" for k in CLASS_NAMES)
        axes[r, 2].set_xlabel(f"MCC {res['mcc']:.3f}   F1: {f1}", fontsize=10)

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
