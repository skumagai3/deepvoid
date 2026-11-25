"""Bolshoi (Klypin, Trujillo-Gomez & Primack 2011).

250 h^-1 Mpc box, 2048^3 particles, WMAP5 cosmology. Pre-built grids (dark
matter, T-web labels, BDM halos at fixed spacings) are listed in
``catalog.yaml``. Halo positions come from CosmoSim's BDM catalog
``bolshoi.bdmv`` at z = 0 (snapshot 416); see notebook
``research/19_bolshoi_halos`` (download).
"""

from __future__ import annotations

import os

import numpy as np

from . import Cosmology, Tracers

BOX_SIZE = 250.0  # h^-1 Mpc
COSMOLOGY = Cosmology(omega_m=0.27, h=0.70)  # WMAP5
SNAPSHOT_Z0 = 416


def read_halos(path: str | os.PathLike, distinct_only: bool = False) -> Tracers:
    """BDM halos from a CosmoSim Parquet download of ``bolshoi.bdmv``.

    Expects columns x, y, z (comoving h^-1 Mpc), vx, vy, vz (peculiar km/s)
    and mvir (h^-1 Msun); names are matched without regard to case. BDM marks
    subhalos with ``hostflag`` = the bdmid of their host (-1 for distinct
    halos). By default hosts and subhalos are both kept, as for TNG and SMDPL;
    ``distinct_only`` keeps hosts only (needs ``hostflag``).
    """
    import pandas as pd

    df = pd.read_parquet(path)
    cols = {c.lower(): c for c in df.columns}
    missing = [c for c in ("x", "y", "z", "vx", "vy", "vz", "mvir") if c not in cols]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; found {list(df.columns)}")
    sub = None
    if "hostflag" in cols:
        sub = df[cols["hostflag"]].to_numpy() != -1
    elif distinct_only:
        raise ValueError(f"{path}: distinct_only needs the hostflag column")
    if distinct_only:
        df = df[~sub]
    get = lambda *names: np.stack([df[cols[n]].to_numpy(np.float64) for n in names], axis=1)
    pos = get("x", "y", "z") % BOX_SIZE
    vel = get("vx", "vy", "vz").astype(np.float32)
    mass = df[cols["mvir"]].to_numpy(np.float64)
    meta = {"sim": "bolshoi", "tracer": "bdm halos" + (" (distinct)" if distinct_only else ""),
            "redshift": 0.0, "cosmology": COSMOLOGY}
    if sub is not None:
        meta["n_subhalos"] = 0 if distinct_only else int(sub.sum())
    return Tracers(pos=pos, box_size=BOX_SIZE, mass=mass, vel=vel, meta=meta)
