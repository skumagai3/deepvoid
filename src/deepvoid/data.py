"""Download the published DeepVoid grids.

The TNG300-3-Dark and Bolshoi grids used in this repository are on Zenodo
(https://doi.org/10.5281/zenodo.24233069). They are derived from IllustrisTNG
and Bolshoi data; see the record for how to credit both simulations.

    from deepvoid import data

    catalog = data.fetch(["tng300-3/dm", "tng300-3/mask"], "/content/deepvoid_data")
    dm = catalog.load("tng300-3/dm")

``fetch`` downloads each file once, checks it against the SHA-256 checksum
in the record's MANIFEST.txt, and writes a ``catalog.yaml`` in the folder,
using the same dataset names as the packaged catalog. A config can then use
the files by setting ``catalog: /content/deepvoid_data/catalog.yaml``.
"""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
from importlib import resources
from pathlib import Path

import yaml

from .catalog import Catalog

RECORD = "24233070"
DOI = "10.5281/zenodo.24233069"   # all versions
URL = "https://zenodo.org/records/{record}/files/{file}?download=1"

# Dataset name (as in the packaged catalog) -> file in the record.
FILES = {
    "tng300-3/dm": "tng300-3-dark_dm_n512.npz",
    "tng300-3/mask": "tng300-3-dark_tweb_n512.npz",
    "tng300-3/subhalos/L3": "tng300-3-dark_subhalos_L3_n512.npz",
    "tng300-3/subhalos/L5": "tng300-3-dark_subhalos_L5_n512.npz",
    "tng300-3/subhalos/L7": "tng300-3-dark_subhalos_L7_n512.npz",
    "tng300-3/subhalos/L10": "tng300-3-dark_subhalos_L10_n512.npz",
    "bolshoi/dm": "bolshoi_dm_n640.npz",
    "bolshoi/mask": "bolshoi_tweb_n640.npz",
    "bolshoi/halos/L3": "bolshoi_halos_L3_n640.npz",
    "bolshoi/halos/L5": "bolshoi_halos_L5_n640.npz",
    "bolshoi/halos/L7": "bolshoi_halos_L7_n640.npz",
    "bolshoi/halos/L10": "bolshoi_halos_L10_n640.npz",
}


def names() -> list[str]:
    """Names of the datasets in the record."""
    return list(FILES)


def fetch(names: list[str] | str | None = None, folder: str | Path = "deepvoid_data",
          url: str = URL) -> Catalog:
    """Download datasets (all of them if ``names`` is None) into ``folder``.

    Files already there with the right checksum are not downloaded again.
    Returns a :class:`Catalog` of everything in the folder.
    """
    names = list(FILES) if names is None else [names] if isinstance(names, str) else list(names)
    unknown = [n for n in names if n not in FILES]
    if unknown:
        raise KeyError(f"not in the record: {unknown}; available: {list(FILES)}")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(folder, url)
    for name in names:
        file = FILES[name]
        path = folder / file
        if path.exists() and _sha256(path) == manifest[file]:
            continue
        print(f"downloading {name} ({file}) ...", flush=True)
        _download(url.format(record=RECORD, file=file), path)
        if _sha256(path) != manifest[file]:
            path.unlink()
            raise OSError(f"{file}: checksum does not match the record's MANIFEST.txt")
    _write_catalog(folder)
    return Catalog(folder / "catalog.yaml")


def _manifest(folder: Path, url: str) -> dict[str, str]:
    path = folder / "MANIFEST.txt"
    _download(url.format(record=RECORD, file="MANIFEST.txt"), path)
    out = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            file, _size, digest, *_ = line.split()
            out[file] = digest
    return out


def _download(url: str, path: Path) -> None:
    tmp = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f, length=1 << 22)
    tmp.replace(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_catalog(folder: Path) -> None:
    """A catalog of the downloaded files, with the packaged catalog's names
    and descriptions."""
    packaged = yaml.safe_load(resources.files("deepvoid").joinpath("catalog.yaml").read_text())
    datasets = {}
    for name, file in FILES.items():
        if (folder / file).exists():
            entry = dict(packaged["datasets"][name])
            entry["file"] = file
            if name.startswith("bolshoi/"):
                # the record's Bolshoi grids are 624^3 grids padded to 640^3 (see catalog.yaml)
                entry["crop"] = 624
                if name == "bolshoi/mask":
                    entry["note"] = ("labels computed on the padded 640^3 grid, cropped; the packaged "
                                     "catalog's 624^3 labels differ on 1.7% of voxels")
                    entry.pop("sigma_eff", None)
            datasets[name] = entry
    text = ("# Grids downloaded from the DeepVoid data release, "
            f"https://doi.org/{DOI}. Paths are relative to this folder.\n"
            + yaml.safe_dump({"simulations": packaged["simulations"], "datasets": datasets},
                             sort_keys=False))
    (folder / "catalog.yaml").write_text(text)
