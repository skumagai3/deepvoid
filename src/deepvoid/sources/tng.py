"""IllustrisTNG (https://www.tng-project.org).

DeepVoid uses TNG300-3-Dark for the dark-matter field and TNG300-1-Dark
subhalos as sparse tracers, at z = 0 (snapshot 99).
"""

from __future__ import annotations

import numpy as np

from . import Cosmology, Tracers

BOX_SIZE = 205.0  # h^-1 Mpc
COSMOLOGY = Cosmology(omega_m=0.3089, h=0.6774)  # Planck 2015


def read_subhalos(path: str, mass_key: str = "SubhaloMass") -> Tracers:
    """Read a subhalo catalog saved as a dict of arrays with ``np.save``.

    Expects the TNG field names (``SubhaloPos`` in ckpc/h, ``SubhaloVel`` in
    km/s, masses in 1e10 h^-1 Msun), as downloaded from the TNG API, and
    converts them to h^-1 Mpc and h^-1 Msun.

    Note: check the key names against the file before relying on this; older
    project files may use different ones.
    """
    cat = np.load(path, allow_pickle=True)[()]
    pos = np.asarray(cat["SubhaloPos"], dtype=np.float64) / 1000.0
    vel = np.asarray(cat["SubhaloVel"], dtype=np.float64) if "SubhaloVel" in cat else None
    mass = np.asarray(cat[mass_key], dtype=np.float64) * 1e10
    return Tracers(pos=pos % BOX_SIZE, box_size=BOX_SIZE, mass=mass, vel=vel,
                   meta={"sim": "TNG300-1-Dark", "tracer": "subhalos", "redshift": 0.0,
                         "cosmology": COSMOLOGY})
