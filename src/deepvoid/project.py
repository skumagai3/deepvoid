"""A project folder: your simulation's grids, their catalog and your runs.

This is the way to train DeepVoid on your own simulation. A project is a
folder with a fixed layout::

    my_project/
      catalog.yaml     names and descriptions of the grids below
      data/            one .npz per grid (density fields and label masks)
      runs/            one folder per training run

Typical use::

    from deepvoid import Project

    project = Project("my_project")
    project.add_density("dm", matter_density, box_size=205.0, tracer="dm")
    project.add_labels("labels", density="dm")            # T-web classes
    project.add_tracers("halos/L5", positions, box_size=205.0, n=512,
                        masses=masses, spacing=5.0)
    cfg = project.config("first_run", inputs=["halos/L5"], labels="labels")

    from deepvoid.train import run
    run(cfg)

Everything is stored on the same grid convention as the rest of the package:
arrays of shape (n, n, n) with axis order (x, y, z), a periodic box, and
lengths in h^-1 Mpc. All grids in one run must have the same n and box size.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .catalog import Catalog
from .config import load_config
from .grid import cic
from .io import save_volume
from .labels import tweb_labels
from .sources import Tracers
from .volume import Volume

CLASS_NAMES = ("void", "wall", "filament", "halo")


class Project:
    """A folder holding grids, their catalog and training runs."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        (self.folder / "data").mkdir(parents=True, exist_ok=True)
        (self.folder / "runs").mkdir(exist_ok=True)
        self.catalog_file = self.folder / "catalog.yaml"
        if not self.catalog_file.exists():
            self._write({})

    def __repr__(self) -> str:
        return f"Project({str(self.folder)!r}, datasets={self.names()})"

    # ---- reading ----

    @property
    def catalog(self) -> Catalog:
        """The project's catalog, re-read from disk."""
        return Catalog(self.catalog_file)

    def names(self) -> list[str]:
        """Names of the grids in the project."""
        return self.catalog.names()

    def load(self, name: str) -> Volume:
        return self.catalog.load(name)

    def summary(self) -> str:
        """One line per grid: size, what it is, and how sparse it is."""
        lines = []
        for name in self.names():
            vol = self.load(name)
            m = vol.meta
            if m.get("field") == "mask":
                frac = np.bincount(vol.data.ravel(), minlength=4) / vol.data.size
                what = "labels: " + ", ".join(f"{c} {100 * f:.1f}%"
                                              for c, f in zip(CLASS_NAMES, frac))
            else:
                what = f"density of {m.get('tracer', 'unknown tracer')}"
                if m.get("spacing") is not None:
                    what += f", mean spacing {m['spacing']:.3g} h^-1 Mpc"
                what += f", {100 * np.mean(vol.data == 0):.1f}% empty cells"
            lines.append(f"{name}: {vol.n}^3 cells of {vol.cell_size:.3g} h^-1 Mpc; {what}")
        return "\n".join(lines) if lines else "(no grids yet)"

    # ---- adding grids ----

    def add_density(self, name: str, density: np.ndarray | Volume, box_size: float | None = None,
                    tracer: str | None = None, spacing: float | None = None,
                    **meta: Any) -> Volume:
        """Add a gridded density field.

        ``density`` is an (n, n, n) array of mass or counts per cell, in any
        units: the network input is rho / rho_bar, computed from the array's
        own mean. It must be non-negative and cover the whole periodic box.

        ``tracer`` says what was gridded (``"dm"``, ``"halos"``,
        ``"galaxies"``, ...) and ``spacing`` is the mean tracer separation
        n^(-1/3) in h^-1 Mpc, where n is the number density. The spacing is
        needed for a smoothed input channel scaled to the spacing and for
        conditioning a model on it.
        """
        vol = _as_volume(density, box_size)
        if not np.all(np.isfinite(vol.data)) or vol.data.min() < 0:
            raise ValueError("a density grid must be finite and non-negative")
        if vol.data.mean() <= 0:
            raise ValueError("the density grid is empty")
        vol.meta.update(field="density", **meta)
        if tracer is not None:
            vol.meta["tracer"] = tracer
        if spacing is not None:
            vol.meta["spacing"] = float(spacing)
        return self._store(name, Volume(vol.data.astype(np.float32), vol.box_size, vol.meta))

    def add_tracers(self, name: str, positions: np.ndarray, box_size: float, n: int,
                    masses: np.ndarray | None = None, spacing: float | None = None,
                    weight_by_mass: bool = True, tracer: str = "halos", **meta: Any) -> Volume:
        """Grid point tracers with cloud-in-cell assignment and add the result.

        ``positions`` is (N, 3) in h^-1 Mpc inside [0, box_size); ``n`` is the
        number of cells per side. With ``spacing`` given, only the most
        massive tracers are kept so that the mean separation is ``spacing``
        (this needs ``masses``); otherwise all tracers are used and the
        spacing is computed as box_size / N^(1/3). ``weight_by_mass`` chooses
        between a mass-weighted grid and plain counts.
        """
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must have shape (N, 3), got {positions.shape}")
        if positions.min() < 0 or positions.max() >= box_size:
            raise ValueError("positions must lie in [0, box_size); wrap them into the box first")
        tracers = Tracers(pos=positions, box_size=float(box_size),
                          mass=None if masses is None else np.asarray(masses, dtype=np.float64),
                          meta={"tracer": tracer, **meta})
        if spacing is not None:
            from .observe.tracers import select_by_spacing
            tracers = select_by_spacing(tracers, spacing)
        else:
            tracers.meta["spacing"] = float(box_size / len(tracers) ** (1 / 3))
        tracers.meta["n_tracers"] = len(tracers)
        weights = "mass" if (weight_by_mass and tracers.mass is not None) else None
        tracers.meta["weights"] = "mass" if weights else "counts"
        return self._store(name, cic(tracers, n, weights=weights))

    def add_labels(self, name: str, density: str, sigma_eff: float = 1.0,
                   threshold: float = 0.65) -> Volume:
        """Compute T-web labels from a density grid already in the project.

        The density should be the full matter field, not sparse tracers: the
        labels are the structure the network is asked to recover. Each cell
        gets the number of tidal-tensor eigenvalues above ``threshold``
        (0 void, 1 wall, 2 filament, 3 halo), after Gaussian smoothing to an
        effective scale ``sigma_eff`` in h^-1 Mpc
        (sigma_eff^2 = sigma^2 + cell size^2). The defaults are the DeepVoid
        convention; the grid cells must be smaller than ``sigma_eff``.
        """
        return self._store(name, tweb_labels(self.load(density), sigma_eff=sigma_eff,
                                             threshold=threshold))

    def add_mask(self, name: str, labels: np.ndarray | Volume, box_size: float | None = None,
                 **meta: Any) -> Volume:
        """Add class labels you computed yourself: integers 0 (void), 1 (wall),
        2 (filament), 3 (halo), one per cell."""
        vol = _as_volume(labels, box_size)
        data = np.asarray(vol.data)
        if not np.issubdtype(data.dtype, np.integer):
            if not np.array_equal(data, np.round(data)):
                raise ValueError("labels must be whole numbers 0 to 3")
        if data.min() < 0 or data.max() > 3:
            raise ValueError("labels must be 0 (void), 1 (wall), 2 (filament) or 3 (halo)")
        vol.meta.update(field="mask", **meta)
        return self._store(name, Volume(data.astype(np.uint8), vol.box_size, vol.meta))

    # ---- configs ----

    def config(self, name: str, inputs: list[str] | str, labels: str, **settings: Any) -> dict:
        """A complete run config for this project, with every setting filled in.

        ``settings`` override the defaults, nested as in a YAML config, e.g.
        ``model={"depth": 5}`` or ``training={"epochs": 40}``. The result can
        be passed to :func:`deepvoid.train.run` or saved with
        :func:`deepvoid.config.save_config`. See ``configs/template.yaml`` for
        what each setting means.
        """
        inputs = [inputs] if isinstance(inputs, str) else list(inputs)
        known = set(self.names())
        missing = [n for n in inputs + [labels] if n not in known]
        if missing:
            raise KeyError(f"not in this project: {missing}; it has {sorted(known)}")
        spec = self._read()
        grids = {n: (spec[n]["n"], spec[n]["box_size"]) for n in inputs + [labels]}
        if len(set(grids.values())) > 1:
            raise ValueError("inputs and labels must share one grid and box size; got "
                             + ", ".join(f"{n}: n={g[0]}, box={g[1]}" for n, g in grids.items()))
        if spec[labels]["field"] != "mask":
            raise ValueError(f"{labels!r} is not a label grid; make one with add_labels or add_mask")
        user = {"name": name, "output": str((self.folder / "runs").resolve()),
                "catalog": str(self.catalog_file.resolve()), **settings}
        data = dict(user.get("data") or {})
        data.update(inputs=inputs, labels=labels)
        user["data"] = data
        return load_config(user)

    # ---- internals ----

    def _store(self, name: str, volume: Volume) -> Volume:
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+(/[A-Za-z0-9_.\-]+)*", name):
            raise ValueError(f"bad dataset name {name!r}: use letters, digits, _ . - and /")
        spec = self._read()
        others = {(e["n"], e["box_size"]) for n, e in spec.items() if n != name}
        if others and (volume.n, volume.box_size) not in others:
            print(f"note: {name!r} is {volume.n}^3 in a {volume.box_size} h^-1 Mpc box, unlike "
                  "the grids already here; grids of different size cannot share a run")
        rel = Path("data") / (name.replace("/", "__") + ".npz")
        volume.meta["name"] = name
        is_mask = volume.meta.get("field") == "mask"
        save_volume(volume, self.folder / rel, dtype="uint8" if is_mask else "float32")
        entry = {"file": str(rel), "field": volume.meta.get("field"),
                 "n": volume.n, "box_size": float(volume.box_size)}
        for key in ("tracer", "spacing"):
            if volume.meta.get(key) is not None:
                entry[key] = volume.meta[key]
        spec[name] = entry
        self._write(spec)
        return volume

    def _read(self) -> dict:
        with open(self.catalog_file) as f:
            return (yaml.safe_load(f) or {}).get("datasets") or {}

    def _write(self, datasets: dict) -> None:
        text = ("# Grids in this project. Paths are relative to this file's folder.\n"
                "# Written by deepvoid.Project; each .npz also carries its own metadata.\n"
                + yaml.safe_dump({"datasets": datasets}, sort_keys=True))
        self.catalog_file.write_text(text)


def _as_volume(data: np.ndarray | Volume, box_size: float | None) -> Volume:
    if isinstance(data, Volume):
        return Volume(data.data, data.box_size, dict(data.meta))
    if box_size is None:
        raise ValueError("box_size (in h^-1 Mpc) is needed with a plain array")
    return Volume(np.asarray(data), float(box_size), {})
