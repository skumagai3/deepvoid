"""Gridding tracers onto a periodic mesh with cloud-in-cell (CIC) assignment."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import ndimage

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


def velocity_field(tracers: Tracers | Iterable[Tracers], n: int, sigma_eff: float = 1.0,
                   sigma_cells: float | None = None,
                   return_count: bool = False) -> list[Volume] | tuple[list[Volume], Volume]:
    """Gaussian-smoothed mean velocity on an n^3 periodic grid.

    Each velocity component (the momentum per unit particle mass) and the
    particle count are assigned with :func:`cic`, both are smoothed with the
    same Gaussian, and the smoothed momentum is divided by the smoothed count.
    All particles count equally (number weighting, as Hoffman et al. 2012; in
    a dark-matter-only run all particles have the same mass).

    Dividing after smoothing keeps the velocity defined in cells no particle
    lands in, which are common in voids when there are only a few particles
    per cell. This is not the same as dividing in each cell and then
    smoothing the velocity (Hoffman et al. 2012, who had ~64 particles per
    cell); the two agree where cells are well populated.

    The smoothing follows the T-web convention: ``sigma_eff`` (h^-1 Mpc) sets
    the Gaussian through sigma_eff^2 = sigma^2 + cell^2
    (:func:`deepvoid.labels.smoothing_sigma`); ``sigma_cells`` overrides it
    with a sigma in cells, e.g. 2.4 to match the existing TNG300-3 mask.

    ``tracers`` may be one :class:`Tracers` with velocities or an iterable of
    them (one per snapshot chunk), so a snapshot can be gridded without
    holding all particles at once.

    Returns [v_x, v_y, v_z] as float32 Volumes in km/s; with ``return_count``
    also the smoothed count per cell. Cells where the smoothed count is not
    positive get velocity 0; how many is recorded as ``meta["empty_cells"]``.
    """
    from .labels.tweb import smoothing_sigma

    if isinstance(tracers, Tracers):
        tracers = [tracers]
    momentum = [None, None, None]
    count = None
    box_size, meta = None, {}
    for t in tracers:
        if t.vel is None:
            raise ValueError("tracers have no velocities")
        if box_size is None:
            box_size, meta = t.box_size, t.meta
        elif t.box_size != box_size:
            raise ValueError("tracer chunks have different box sizes")
        c = cic(t, n, weights=None).data
        count = c if count is None else count + c
        for i in range(3):
            m = cic(t, n, weights=t.vel[:, i]).data
            momentum[i] = m if momentum[i] is None else momentum[i] + m
    if count is None:
        raise ValueError("no tracers given")

    cell = box_size / n
    if sigma_cells is None:
        sigma_cells = smoothing_sigma(box_size, n, sigma_eff)
    count = ndimage.gaussian_filter(count, sigma_cells, mode="wrap")
    filled = count > 0
    meta = {k: v for k, v in meta.items() if k not in ("cosmology", "ids")}
    meta.update(field="velocity", units="km/s", assignment="cic", weighting="number",
                sigma_cells=float(sigma_cells),
                sigma_eff=float(np.hypot(sigma_cells * cell, cell)),
                empty_cells=int((~filled).sum()))
    velocity = []
    for i, axis in enumerate("xyz"):
        m = ndimage.gaussian_filter(momentum[i], sigma_cells, mode="wrap")
        v = np.zeros_like(m)
        np.divide(m, count, out=v, where=filled)
        velocity.append(Volume(data=v.astype(np.float32), box_size=box_size,
                               meta={**meta, "component": axis}))
    if return_count:
        cmeta = {**meta, "field": "count"}
        cmeta.pop("units")
        return velocity, Volume(data=count.astype(np.float32), box_size=box_size, meta=cmeta)
    return velocity
