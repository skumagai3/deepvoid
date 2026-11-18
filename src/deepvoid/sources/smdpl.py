"""SMDPL, the Small MultiDark Planck simulation (Klypin et al. 2016).

400 h^-1 Mpc box, 3840^3 particles, Planck cosmology. Halos come from the
Rockstar catalog on CosmoSim (www.cosmosim.org); the dark-matter density at
z = 0 from CosmoSim's ``smdpl.dens512_z0`` table (512^3 cells, values are
rho / rho_bar). See notebooks ``research/15_smdpl`` (download) and
``research/16_smdpl_scoring``.
"""

from __future__ import annotations

import os

import numpy as np

from . import Cosmology, Tracers

BOX_SIZE = 400.0  # h^-1 Mpc
COSMOLOGY = Cosmology(omega_m=0.307115, h=0.6777)  # Planck 2013


def read_halos(path: str | os.PathLike) -> Tracers:
    """Rockstar halos and subhalos from a CosmoSim Parquet download.

    Expects columns x, y, z (comoving h^-1 Mpc), vx, vy, vz (peculiar km/s)
    and mvir (h^-1 Msun); column names are matched without regard to case.
    Hosts and subhalos are both kept, as TNG's Subfind subhalos are; the
    ``upid`` column, if present, goes into ``meta['n_subhalos']`` only.
    """
    import pandas as pd

    df = pd.read_parquet(path)
    cols = {c.lower(): c for c in df.columns}
    missing = [c for c in ("x", "y", "z", "vx", "vy", "vz", "mvir") if c not in cols]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; found {list(df.columns)}")
    get = lambda *names: np.stack([df[cols[n]].to_numpy(np.float64) for n in names], axis=1)
    pos = get("x", "y", "z") % BOX_SIZE
    vel = get("vx", "vy", "vz").astype(np.float32)
    mass = df[cols["mvir"]].to_numpy(np.float64)
    meta = {"sim": "smdpl", "tracer": "rockstar halos", "redshift": 0.0,
            "cosmology": COSMOLOGY}
    if "upid" in cols:
        meta["n_subhalos"] = int((df[cols["upid"]].to_numpy() != -1).sum())
    return Tracers(pos=pos, box_size=BOX_SIZE, mass=mass, vel=vel, meta=meta)
