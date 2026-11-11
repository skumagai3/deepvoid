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
