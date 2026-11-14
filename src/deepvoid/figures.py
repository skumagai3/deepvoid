"""Standard figures.

All functions take NumPy arrays (or :class:`deepvoid.runs.Run` objects
where noted) and return a Matplotlib figure; nothing is saved unless you
call :func:`save_and_display` or ``fig.savefig``.

- :func:`fields_figure`: slices of density fields and a T-web mask
- :func:`training_curves`: validation scores and loss by epoch, for runs
- :func:`prediction_figure`: input, truth, prediction, void score and errors
- :func:`summary_figure`: the one-figure overview of what DeepVoid does
- :func:`evaluation_figure`, :func:`void_pr_figure`: used by evaluation

:func:`evaluation_figure` is drawn by ``train.evaluate_run`` for every
scored run: for each input, a slice through the middle of the test slab
(input, truth, prediction, wrong voxels), the confusion matrix as fractions
of each true class, void precision, recall and MCC on the test slab against
the void-score threshold, and the reliability of the void score. The
prediction and confusion matrix use the highest-score class (argmax).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .labels import CLASS_NAMES

MASK_COLORS = ["#2b6cb0", "#9ae6b4", "#f6ad55", "#c53030"]   # void, wall, filament, halo


# ---------------------------------------------------------------- building blocks

def take_slice(volume: np.ndarray, index: int, axis: int = 0, thickness: int = 1) -> np.ndarray:
    """A 2D slice; with ``thickness`` > 1, the mean over that many cells (periodic)."""
    n = volume.shape[axis]
    idx = np.arange(index - thickness // 2, index - thickness // 2 + thickness) % n
    return np.take(volume, idx, axis=axis).mean(axis=axis)


def log_density(slice2d: np.ndarray, floor: float = 1e-2) -> np.ndarray:
    """log10(rho / rho_mean + floor) of a density slice, for display."""
    mean = slice2d.mean()
    return np.log10(slice2d / (mean if mean > 0 else 1.0) + floor)


def show_density(ax, slice2d, box_size=None, title=None, vmin=None, vmax=None):
    """Plot log10(rho / rho_mean + 0.01); colour range from the non-empty cells."""
    img = log_density(slice2d)
    filled = img[img > np.log10(1e-2) + 1e-6]
    lo = np.log10(1e-2) if vmin is None else vmin
    hi = (np.percentile(filled, 99.5) if filled.size else 1.0) if vmax is None else vmax
    ext = [0, box_size, 0, box_size] if box_size else None
    im = ax.imshow(img.T, origin="lower", extent=ext, cmap="magma", vmin=lo, vmax=hi)
    if title:
        ax.set_title(title)
    return im


def show_mask(ax, slice2d, box_size=None, title=None):
    """Plot T-web classes (or predicted classes) with the standard colours."""
    from matplotlib.colors import ListedColormap
    ext = [0, box_size, 0, box_size] if box_size else None
    im = ax.imshow(slice2d.T, origin="lower", extent=ext, cmap=ListedColormap(MASK_COLORS),
                   vmin=-0.5, vmax=3.5, interpolation="nearest")
    if title:
        ax.set_title(title)
    return im


def mask_legend(fig_or_ax, **kwargs):
    """A legend naming the four classes by colour."""
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=n) for c, n in zip(MASK_COLORS, CLASS_NAMES)]
    return fig_or_ax.legend(handles=handles, **kwargs)


def _label_axes(ax, box_size):
    if box_size:
        ax.set_xlabel("[h⁻¹ Mpc]")
    else:
        ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------- figures

def fields_figure(fields: dict[str, np.ndarray], mask: np.ndarray | None, index: int,
                  axis: int = 0, box_size: float | None = None, thickness: int = 1,
                  title: str | None = None):
    """One panel per density field (log scale), then the mask, all at the same slice.

    ``fields`` maps a panel title to an (n, n, n) density array; fields on a
    coarser grid are sliced at the matching position.
    """
    import matplotlib.pyplot as plt
    n_panels = len(fields) + (mask is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 4.8), squeeze=False)
    n_ref = mask.shape[0] if mask is not None else next(iter(fields.values())).shape[0]
    for ax, (label, field) in zip(axes[0], fields.items()):
        f = field.shape[0] / n_ref
        show_density(ax, take_slice(field, int(index * f), axis, max(1, int(thickness * f))),
                     box_size, label)
        _label_axes(ax, box_size)
    if mask is not None:
        show_mask(axes[0, -1], take_slice(mask, index, axis), box_size, "T-web classes")
        _label_axes(axes[0, -1], box_size)
        mask_legend(axes[0, -1], loc="upper right", fontsize=8, framealpha=0.9)
    where = f"slice {'xyz'[axis]} = {index}" + (f", {thickness} cells thick" if thickness > 1 else "")
    fig.suptitle(title or f"Density fields and T-web classes ({where}; colour: log10(ρ/ρ̄ + 0.01))",
                 y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def training_curves(runs, keys=("val_void_mcc", "val_mcc", "val_loss"), title=None):
    """Validation scores (and loss) by epoch, one line per run.

    ``runs`` is a list of :class:`deepvoid.runs.Run`. Runs logged before
    ``val_void_mcc`` existed are skipped in that panel.
    """
    import matplotlib.pyplot as plt
    names = {"val_void_mcc": "void MCC (void vs rest)", "val_mcc": "MCC (four classes)",
             "val_loss": "loss (cross-entropy)", "loss": "training loss"}
    fig, axes = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 4.2), squeeze=False)
    for ax, key in zip(axes[0], keys):
        for run in runs:
            log = run.log
            if key in log:
                ax.plot(log["epoch"], log[key], label=run.name, lw=1.4)
        ax.set_title(f"validation {names.get(key, key)}" if key.startswith("val_") else names.get(key, key))
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title or "Training progress (validation slab, fixed crops)", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def prediction_figure(inp: np.ndarray, truth: np.ndarray, scores: np.ndarray, index: int,
                      axis: int = 0, threshold: float | None = None,
                      box_size: float | None = None, title: str | None = None,
                      input_label: str = "input"):
    """Input, truth, predicted classes, void score and misclassified voxels at one slice.

    ``inp`` is the normalized input (n, n, n) or (n, n, n, channels; the last
    channel is shown), ``scores`` the class scores (n, n, n, 4) on the same
    grid as ``truth``. With ``threshold``, voxels with void score >= threshold
    are called void.
    """
    import matplotlib.pyplot as plt
    from .evaluate import decide
    pred = decide(take_slice_scores(scores, index, axis), threshold)
    tru = take_slice(truth, index, axis).astype(int)
    x = inp[..., -1] if inp.ndim == 4 else inp
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    ext = [0, box_size, 0, box_size] if box_size else None
    img = take_slice(x, index, axis)
    lo, hi = np.percentile(img, [1, 99.7])
    axes[0].imshow(img.T, origin="lower", extent=ext, cmap="magma", vmin=lo,
                   vmax=hi if hi > lo else img.max())
    axes[0].set_title(input_label)
    show_mask(axes[1], tru, box_size, "T-web classes (truth)")
    show_mask(axes[2], pred, box_size,
              "predicted classes" + (f" (void if score ≥ {threshold:.2f})" if threshold else ""))
    im = axes[3].imshow(take_slice_scores(scores, index, axis)[..., 0].T, origin="lower",
                        extent=ext, cmap="Blues", vmin=0, vmax=1)
    axes[3].set_title("void score")
    fig.colorbar(im, ax=axes[3], fraction=0.046)
    show_mask(axes[4], np.where(pred != tru, tru, np.nan), box_size,
              "misclassified voxels, coloured by true class")
    axes[4].set_facecolor("#eeeeee")
    for ax in axes:
        _label_axes(ax, box_size)
    fig.suptitle(title or f"Prediction at slice {'xyz'[axis]} = {index}", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    mask_legend(fig, loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    return fig


def take_slice_scores(scores: np.ndarray, index: int, axis: int = 0) -> np.ndarray:
    """A 2D slice of class scores (n, n, n, 4) -> (n, n, 4)."""
    return np.take(scores, index, axis=axis)


def summary_figure(dm: np.ndarray, truth: np.ndarray, tracers: np.ndarray,
                   pred: np.ndarray, index: int, axis: int = 0,
                   box_size: float | None = None, spacing: float | None = None,
                   tracer_depth: int = 25, note: str | None = None):
    """The DeepVoid idea in one figure.

    Top row, the simulation: dark-matter density and the T-web classes
    computed from it (the target). Bottom row, the observation-like side:
    the sparse tracers (projected through a slab ``tracer_depth`` cells
    thick, centred on the slice) and the classes DeepVoid predicts from them.
    ``tracers`` is the gridded tracer field (non-zero where there are
    tracers); ``pred`` the predicted classes (n, n, n).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    fig, axes = plt.subplots(2, 2, figsize=(11, 12.6))
    ext_box = box_size or truth.shape[0]
    show_density(axes[0, 0], take_slice(dm, index, axis, thickness=2), box_size)
    axes[0, 0].set_title("Dark matter (simulation)")
    show_mask(axes[0, 1], take_slice(truth, index, axis), box_size)
    axes[0, 1].set_title("T-web classes (target)")

    proj = take_slice(tracers > 0, index, axis, thickness=tracer_depth) > 0
    ii, jj = np.nonzero(proj)
    scale = ext_box / truth.shape[0]
    ax = axes[1, 0]
    ax.set_facecolor("black")
    ax.scatter((ii + 0.5) * scale, (jj + 0.5) * scale, s=5, c="#fbd38d", lw=0)
    ax.set_xlim(0, ext_box); ax.set_ylim(0, ext_box); ax.set_aspect("equal")
    depth = tracer_depth * scale
    sp = f", spacing {spacing:g} h⁻¹ Mpc" if spacing else ""
    ax.set_title(f"Sparse tracers (input{sp})\nprojected through a {depth:.0f} h⁻¹ Mpc thick slab")
    show_mask(axes[1, 1], take_slice(pred, index, axis), box_size)
    axes[1, 1].set_title("DeepVoid prediction from the tracers")
    for a in axes.ravel():
        _label_axes(a, box_size)
        a.title.set_position((0.5, 1.0)); a.set_title(a.get_title(), pad=10)

    fig.suptitle("DeepVoid: cosmic-web classes from sparse tracers", fontsize=15, y=0.995)
    # Lay the panels out first, then put each arrow level with its row.
    fig.tight_layout(rect=(0, 0.085, 1, 0.97), h_pad=5.0, w_pad=7.0)
    arrow = dict(arrowstyle="-|>", mutation_scale=22, lw=2, color="#4a5568")
    for row, text in ((0, "T-web"), (1, "DeepVoid\n(3D U-Net)")):
        left, right = axes[row, 0].get_position(), axes[row, 1].get_position()
        y = (left.y0 + left.y1) / 2
        gap = right.x0 - left.x1
        x0, x1 = left.x1 + 0.12 * gap, left.x1 + 0.6 * gap    # clear of the tick labels
        fig.patches.append(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, **arrow))
        fig.text((x0 + x1) / 2, y + 0.022, text, ha="center", fontsize=10, color="#4a5568")
    mask_legend(fig, loc="lower center", ncol=4, fontsize=11, bbox_to_anchor=(0.5, -0.01))
    if note:
        fig.text(0.5, 0.052, note, ha="center", fontsize=10)
    return fig


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
        res = row["result"]                     # argmax scores
        tuned = res.get("tuned", {}).get("void_threshold")
        n = row["truth"].shape[0]
        ext = [0, box_size, 0, box_size] if box_size else [0, n, 0, n]
        label = row["name"].rsplit("/", 1)[-1]
        img = row["input"]
        lo, hi = np.percentile(img, [1, 99.7])
        if hi <= lo:                                  # a sparse raw field is mostly one value
            hi = img.max() if img.max() > lo else lo + 1
        axes[r, 0].imshow(img.T, origin="lower", extent=ext, cmap="magma", vmin=lo, vmax=hi)
        axes[r, 0].set_title(f"{label} input: {row.get('input_label', 'input')}")
        pred_title = "predicted classes (highest score)"
        for c, (key, name) in enumerate((("truth", "truth"), ("pred", pred_title)), start=1):
            axes[r, c].imshow(row[key].T, origin="lower", extent=ext, cmap=cmap,
                              vmin=-0.5, vmax=3.5, interpolation="nearest")
            axes[r, c].set_title(name)
        wrong = np.where(row["pred"] != row["truth"], row["truth"], np.nan)
        axes[r, 3].imshow(wrong.T, origin="lower", extent=ext, cmap=cmap, vmin=-0.5, vmax=3.5,
                          interpolation="nearest")
        axes[r, 3].set_facecolor("#eeeeee")
        axes[r, 3].set_title("misclassified voxels, coloured by true class")
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
        ax.set_title("confusion matrix (rows sum to 1)")

        curve = row["result"].get("void_curve", {}).get("test")
        if curves and curve:
            ax = axes[r, 5]
            t = np.asarray(curve["thresholds"])
            for key, color in (("precision", "#2b6cb0"), ("recall", "#dd6b20"), ("mcc", "k")):
                ax.plot(t, curve[key], color=color, label=f"void {key}")
            if tuned is not None:
                ax.axvline(tuned, color="grey", ls="--", lw=1,
                           label=f"tuned threshold {tuned:.2f} (optional)")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
            ax.set_xlabel("threshold on void score")
            ax.set_title("void vs rest: scores against threshold")
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
            ax.set_title(f"void-score calibration (ECE {rel['ece']:.3f})")
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


def void_pr_figure(curves: dict[str, dict[str, dict]], points: dict | None = None,
                   title: str = "Void precision against recall on the test slab"):
    """Void precision (purity) against recall (completeness), one panel per spacing.

    ``curves`` maps run name -> spacing label -> curve (``precision`` and
    ``recall`` at each threshold). ``points`` optionally maps run name ->
    spacing label -> (recall, precision) of an operating point to mark,
    e.g. the argmax prediction's. The legend gives each curve's average
    precision (area under precision against recall).
    """
    import matplotlib.pyplot as plt

    from .evaluate import average_precision

    labels = sorted({L for c in curves.values() for L in c}, key=lambda s: int(s.lstrip("L")))
    fig, axes = plt.subplots(1, len(labels), figsize=(5.2 * len(labels), 5), squeeze=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, L in zip(axes[0], labels):
        for i, (run, by_label) in enumerate(curves.items()):
            if L not in by_label:
                continue
            c = by_label[L]
            color = colors[i % len(colors)]
            ax.plot(c["recall"], c["precision"], "-", color=color, lw=1.5,
                    label=f"{run} (AP {average_precision(c):.3f})")
            if points and L in points.get(run, {}):
                rec, prec = points[run][L]
                ax.plot(rec, prec, "o", color=color, ms=6)
        ax.set_xlim(0.48, 1.01); ax.set_ylim(0.48, 1.01); ax.grid(alpha=0.3)
        ax.set_xlabel("void recall (completeness)")
        ax.set_ylabel("void precision (purity)")
        ax.set_title(f"tracer spacing {L.lstrip('L')} h⁻¹ Mpc")
    for ax in axes[0]:
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(f"{title}\n(dots: each run's predictions, highest-score class)", y=1.03)
    fig.tight_layout()
    return fig

