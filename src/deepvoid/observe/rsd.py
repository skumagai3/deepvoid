"""Redshift-space distortions in the plane-parallel (distant-observer) limit."""

from __future__ import annotations

import numpy as np

from ..sources import Cosmology, Tracers


def to_redshift_space(tracers: Tracers, axis: int = 2,
                      cosmology: Cosmology | None = None) -> Tracers:
    """Move tracers along one axis by their peculiar velocity.

    s = x + v_los (1 + z) / H(z), with H(z) = 100 E(z) h km/s/Mpc, so
    positions stay in h^-1 Mpc. Positions wrap around the periodic box.

    ``tracers.vel`` must be the peculiar velocity in km/s. (TNG particle
    snapshots store velocities that need a factor sqrt(a); subhalo catalogs
    are already peculiar.)
    """
    if tracers.vel is None:
        raise ValueError("redshift-space distortions need tracer velocities")
    cosmology = cosmology or tracers.meta.get("cosmology")
    if cosmology is None:
        raise ValueError("pass a cosmology or set tracers.meta['cosmology']")
    z = tracers.meta.get("redshift", 0.0)

    shift = tracers.vel[:, axis] * (1 + z) / (100.0 * cosmology.hubble_e(z))
    pos = tracers.pos.copy()
    pos[:, axis] = (pos[:, axis] + shift) % tracers.box_size

    out = Tracers(pos=pos, box_size=tracers.box_size, mass=tracers.mass,
                  vel=tracers.vel, meta=dict(tracers.meta))
    out.meta["rsd"] = True
    out.meta["los_axis"] = axis
    return out


def los_to_z(array: np.ndarray, los_axis: int) -> np.ndarray:
    """The array with its line of sight moved to axis 2 (z), by swapping two axes.

    ``los_axis`` 2 returns the array unchanged; 0 or 1 swaps that axis with
    axis 2. Any further axes (channels) are left alone. A swap undoes itself,
    so the same call moves a prediction made in the swapped frame back to the
    original one. Returns a view; use ``np.ascontiguousarray`` for a copy.
    """
    if los_axis not in (0, 1, 2):
        raise ValueError(f"los_axis must be 0, 1 or 2, not {los_axis}")
    return array if los_axis == 2 else np.swapaxes(array, los_axis, 2)
