"""IllustrisTNG (https://www.tng-project.org).

DeepVoid uses TNG300-3-Dark for the dark-matter field and TNG300-1 subhalos
as sparse tracers, at z = 0 (snapshot 99).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import Cosmology, Tracers

BOX_SIZE = 205.0  # h^-1 Mpc
COSMOLOGY = Cosmology(omega_m=0.3089, h=0.6774)  # Planck 2015
API = "https://www.tng-project.org/api"
SUBHALO_FIELDS = ("SubhaloPos", "SubhaloMass", "SubhaloVel")


def fetch_subhalos(path: str | os.PathLike, api_key: str, sim: str = "TNG300-1",
                   snapshot: int = 99, fields: tuple[str, ...] = SUBHALO_FIELDS,
                   tmp_dir: str | os.PathLike = "/tmp") -> Path:
    """Download selected subhalo fields of a TNG group catalog and save them as .npz.

    Asks the TNG web API for the whole catalog restricted to ``fields``
    (a few hundred MB for TNG300-1), and falls back to the catalog's file
    chunks one at a time if the server refuses the full request. Values are
    saved in TNG units (positions in ckpc/h, masses in 1e10 h^-1 Msun,
    velocities in km/s), with the simulation name and snapshot, so
    :func:`read_subhalos` can read the file.

    The API key is on your profile page at tng-project.org.
    """
    import h5py
    import requests

    path = Path(path)
    headers = {"api-key": api_key}
    query = {"Subhalo": ",".join(fields)}
    base = f"{API}/{sim}/files/groupcat-{snapshot}"

    def download(url: str, dest: Path) -> bool:
        with requests.get(url, headers=headers, params=query, stream=True, timeout=600) as r:
            if r.status_code != 200:
                return False
            with open(dest, "wb") as f:
                for block in r.iter_content(chunk_size=1 << 22):
                    f.write(block)
        return True

    tmp = Path(tmp_dir) / f"{sim}_groupcat_{snapshot}.hdf5"
    if download(base + "/", tmp):
        files = [tmp]
    else:
        files, i = [], 0
        while True:
            chunk = Path(tmp_dir) / f"{sim}_groupcat_{snapshot}.{i}.hdf5"
            if not download(f"{base}.{i}.hdf5", chunk):
                break
            files.append(chunk)
            i += 1
        if not files:
            raise RuntimeError(f"could not download the {sim} group catalog; check the API key")

    parts = {k: [] for k in fields}
    for f in files:
        with h5py.File(f, "r") as h:
            if "Subhalo" not in h:   # chunks holding only FoF groups
                continue
            for k in fields:
                parts[k].append(h["Subhalo"][k][:])
        f.unlink()
    data = {k: np.concatenate(v) for k, v in parts.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, sim=sim, snapshot=snapshot, **data)
    return path


def read_subhalos(path: str | os.PathLike, mass_key: str = "SubhaloMass",
                  sim: str | None = None) -> Tracers:
    """Read a subhalo catalog saved by :func:`fetch_subhalos` or with ``np.save``.

    Expects the TNG field names (``SubhaloPos`` in ckpc/h, ``SubhaloVel`` in
    km/s, masses in 1e10 h^-1 Msun), as downloaded from the TNG API, and
    converts them to h^-1 Mpc and h^-1 Msun. Older project files are pickled
    dicts saved with ``np.save``; check their key names before relying on them.

    ``sim`` names the simulation in the returned metadata. Files from
    :func:`fetch_subhalos` record it themselves; older files don't, so pass it.
    """
    path = Path(path)
    if path.suffix == ".npz":
        cat = dict(np.load(path))
        sim = sim or str(cat.get("sim", "unknown"))
    else:
        cat = np.load(path, allow_pickle=True)[()]
        sim = sim or "unknown"
    pos = np.asarray(cat["SubhaloPos"], dtype=np.float64) / 1000.0
    vel = np.asarray(cat["SubhaloVel"], dtype=np.float64) if "SubhaloVel" in cat else None
    mass = np.asarray(cat[mass_key], dtype=np.float64) * 1e10
    return Tracers(pos=pos % BOX_SIZE, box_size=BOX_SIZE, mass=mass, vel=vel,
                   meta={"sim": sim, "tracer": "subhalos", "redshift": 0.0,
                         "cosmology": COSMOLOGY})
