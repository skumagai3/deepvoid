"""Gridding tracers onto a periodic mesh with cloud-in-cell (CIC) assignment."""

from __future__ import annotations

import numpy as np

from .sources import Tracers
from .volume import Volume


def cic(tracers: Tracers, n: int, weights: str | np.ndarray | None = "mass",
        chunk: int = 10_000_000) -> Volume:
    """Assign tracers to an n^3 periodic grid with CIC.

    Grid point (i, j, k) sits at position (i, j, k) * cell_size, and each
    tracer is shared among the 8 surrounding grid points by the overlap of a
    cell-sized cube centered on it. Total weight is conserved.

    Parameters
    ----------
    weights:
        ``"mass"`` to weight by tracer mass, ``None`` for equal weights, or an
        array with one weight per tracer (for example a luminosity or color).
    chunk:
        Tracers processed per pass, to bound memory for large particle sets.

    Returns a float32 :class:`Volume` holding the summed weight per cell.
    Divide by its mean to get rho / rho_bar.
    """
    if isinstance(weights, str):
        if weights != "mass":
            raise ValueError("weights must be 'mass', None or an array")
        w_all = tracers.mass
    else:
        w_all = weights

    cell = tracers.box_size / n
    grid = np.zeros(n ** 3, dtype=np.float64)
    for start in range(0, len(tracers), chunk):
        stop = start + chunk
        x = tracers.pos[start:stop] / cell
        w = np.ones(len(x)) if w_all is None else np.asarray(w_all[start:stop], float)
        i0 = np.floor(x).astype(np.int64)
        d = x - i0
        for dx in (0, 1):
            wx = d[:, 0] if dx else 1 - d[:, 0]
            ix = (i0[:, 0] + dx) % n
            for dy in (0, 1):
                wy = d[:, 1] if dy else 1 - d[:, 1]
                iy = (i0[:, 1] + dy) % n
                for dz in (0, 1):
                    wz = d[:, 2] if dz else 1 - d[:, 2]
                    iz = (i0[:, 2] + dz) % n
                    flat = (ix * n + iy) * n + iz
                    grid += np.bincount(flat, weights=w * wx * wy * wz, minlength=n ** 3)

    meta = {k: v for k, v in tracers.meta.items() if k != "cosmology"}
    meta.update(field="density", assignment="cic")
    return Volume(data=grid.reshape(n, n, n).astype(np.float32),
                  box_size=tracers.box_size, meta=meta)
