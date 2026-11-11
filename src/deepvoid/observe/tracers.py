"""Tracer selection."""

from __future__ import annotations

import numpy as np

from ..sources import Tracers


def select_by_spacing(tracers: Tracers, spacing: float) -> Tracers:
    """Keep the most massive tracers so that the mean separation is ``spacing``.

    The number kept is N = (box_size / spacing)^3, the convention used in the
    DeepVoid papers (lambda = n^(-1/3)).
    """
    if tracers.mass is None:
        raise ValueError("selection by spacing needs tracer masses")
    n_keep = round((tracers.box_size / spacing) ** 3)
    if n_keep > len(tracers):
        raise ValueError(f"spacing {spacing} needs {n_keep} tracers, only {len(tracers)} exist")
    keep = np.argsort(tracers.mass)[-n_keep:]
    out = tracers.subset(keep)
    out.meta["spacing"] = spacing
    out.meta["selection"] = "top mass"
    return out
