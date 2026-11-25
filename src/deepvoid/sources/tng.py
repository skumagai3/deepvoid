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
DM_FIELDS = ("Coordinates", "Velocities")


def fetch_subhalos(path: str | os.PathLike, api_key: str, sim: str = "TNG300-1",
                   snapshot: int = 99, fields: tuple[str, ...] = SUBHALO_FIELDS,
                   tmp_dir: str | os.PathLike = "/tmp", attempts: int = 15,
                   wait: float = 60.0) -> Path:
    """Download selected subhalo fields of a TNG group catalog and save them as .npz.

    The TNG web API serves one field of the whole catalog per request, so
    each field is a separate download (about 170 MB per 3-vector field for
    TNG300-1). The server assembles a field file on the first request, which
    for a large catalog can outlast the gateway's timeout (HTTP 502/503/504);
    the request is then repeated every ``wait`` seconds, up to ``attempts``
    times, until the finished file is served. Fields already downloaded are
    kept in ``tmp_dir``, so calling again resumes.

    Values are saved in TNG units (positions in ckpc/h, masses in
    1e10 h^-1 Msun, velocities in km/s), with the simulation name and
    snapshot, so :func:`read_subhalos` can read the file. The API key is on
    your profile page at tng-project.org.
    """
    import time

    import h5py
    import requests

    path = Path(path)
    url = f"{API}/{sim}/files/groupcat-{snapshot}/"
    data = {}
    for field in fields:
        done = Path(tmp_dir) / f"{sim}_snap{snapshot}_{field}.npy"
        if done.exists():
            data[field] = np.load(done)
            continue
        tmp = done.with_suffix(".hdf5")
        for attempt in range(1, attempts + 1):
            with requests.get(url, headers={"api-key": api_key}, params={"Subhalo": field},
                              stream=True, timeout=900) as r:
                if r.status_code == 200:
                    with open(tmp, "wb") as f:
                        for block in r.iter_content(chunk_size=1 << 22):
                            f.write(block)
                    break
                if r.status_code not in (502, 503, 504) or attempt == attempts:
                    raise RuntimeError(f"{r.url} returned HTTP {r.status_code}: {r.text[:200]}")
            print(f"{field}: server still preparing the file (HTTP {r.status_code}), "
                  f"retrying in {wait:.0f} s ({attempt}/{attempts})")
            time.sleep(wait)
        with h5py.File(tmp, "r") as h:
            data[field] = h["Subhalo"][field][:]
        np.save(done, data[field])
        tmp.unlink()
        print(f"{field}: {data[field].shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, sim=sim, snapshot=snapshot, **data)
    for field in fields:
        (Path(tmp_dir) / f"{sim}_snap{snapshot}_{field}.npy").unlink(missing_ok=True)
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


def fetch_snapshot_dm(out_dir: str | os.PathLike, api_key: str, sim: str = "TNG300-3-Dark",
                      snapshot: int = 99, fields: tuple[str, ...] = DM_FIELDS,
                      chunks: int | None = None, attempts: int = 15,
                      wait: float = 60.0) -> list[Path]:
    """Download the dark-matter fields of a TNG snapshot, one chunk file at a time.

    Each chunk is requested with only ``fields`` of PartType1 (the API's
    ``?dm=...`` subset), so only positions and velocities come down unless
    more are asked for (``ParticleIDs`` is needed to match particles to the
    initial conditions). Chunks already on disk are skipped, so calling again
    resumes. A failed request is repeated every ``wait`` seconds, up to
    ``attempts`` times. ``chunks`` defaults to ``num_files_snapshot`` from the
    API's simulation record (4 for TNG300-3-Dark).

    Returns the chunk files, which :func:`read_snapshot_dm` reads.
    """
    import time

    import requests

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = {"api-key": api_key}
    if chunks is None:
        r = requests.get(f"{API}/{sim}/", headers=headers, timeout=60)
        r.raise_for_status()
        chunks = int(r.json()["num_files_snapshot"])
    files = []
    for chunk in range(chunks):
        path = out_dir / f"snap_{snapshot:03d}.{chunk}.hdf5"
        files.append(path)
        if path.exists() and _complete(path, fields):
            continue
        tmp = path.with_suffix(".part")
        url = f"{API}/{sim}/files/snapshot-{snapshot}.{chunk}.hdf5"
        for attempt in range(1, attempts + 1):
            try:
                with requests.get(url, headers=headers, params={"dm": ",".join(fields)},
                                  stream=True, timeout=900) as r:
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        for block in r.iter_content(chunk_size=1 << 22):
                            f.write(block)
                tmp.replace(path)
                if _complete(path, fields):
                    break
                message = "file incomplete"
            except (requests.RequestException, OSError) as err:
                message = str(err)[:200]
            if attempt == attempts:
                raise RuntimeError(f"{url}: gave up after {attempts} attempts ({message})")
            print(f"chunk {chunk}: {message}; retrying in {wait:.0f} s ({attempt}/{attempts})")
            time.sleep(wait)
        print(f"chunk {chunk}: {path.stat().st_size / 1e9:.2f} GB")
    return files


def _complete(path: Path, fields: tuple[str, ...]) -> bool:
    """True if a chunk file opens and holds every requested field in full."""
    import h5py

    try:
        with h5py.File(path, "r") as h:
            n = int(h["Header"].attrs["NumPart_ThisFile"][1])
            return all(h["PartType1"][f].shape[0] == n for f in fields)
    except (OSError, KeyError):
        return False


def snapshot_files(path: str | os.PathLike) -> list[Path]:
    """The chunk files of one snapshot, in chunk order.

    ``path`` is one chunk file, or a directory holding them: the TNG Lab's
    ``.../output/snapdir_099`` or the folder :func:`fetch_snapshot_dm` wrote.
    """
    path = Path(path)
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.hdf5"), key=lambda p: int(p.suffixes[-2].lstrip(".")))
    if not files:
        raise FileNotFoundError(f"no .hdf5 chunk files in {path}")
    return files


def read_snapshot_dm(path: str | os.PathLike, sim: str = "TNG300-3-Dark",
                     ids: bool = False) -> Tracers:
    """Dark-matter particles of a TNG snapshot, from one chunk file or a whole directory.

    Converts TNG units to the package's: Coordinates (comoving ckpc/h) to
    h^-1 Mpc, and Velocities (km sqrt(a)/s) to peculiar velocities in km/s by
    multiplying by sqrt(a) (TNG data specifications; a = 1 at snapshot 99).
    Positions and velocities are float32 (about 5.9 GB for the 625^3
    particles of TNG300-3-Dark); pass a single chunk file to read less at
    once. With ``ids``, ``meta["ids"]`` holds the ParticleIDs (uint64).
    """
    import h5py

    pos, vel, pid = [], [], []
    for f in snapshot_files(path):
        with h5py.File(f, "r") as h:
            head = h["Header"].attrs
            box = float(head["BoxSize"]) / 1000.0
            a = float(head["Time"])
            mass = float(head["MassTable"][1]) * 1e10
            dm = h["PartType1"]
            pos.append((dm["Coordinates"][:] / 1000.0).astype(np.float32))
            vel.append((dm["Velocities"][:] * np.sqrt(a)).astype(np.float32))
            if ids:
                pid.append(dm["ParticleIDs"][:])
    pos = np.concatenate(pos) % np.float32(box)
    meta = {"sim": sim, "tracer": "dm", "redshift": 1.0 / a - 1.0, "cosmology": COSMOLOGY,
            "particle_mass": mass}
    if ids:
        meta["ids"] = np.concatenate(pid)
    return Tracers(pos=pos, box_size=box, mass=None, vel=np.concatenate(vel), meta=meta)
