"""Named datasets.

Notebooks and configs refer to data by name (``"tng300-3/dm"``) instead of by
file path. ``catalog.yaml`` (shipped with the package; pass another path to
use your own) maps each name to a file under a data root and
records what the file is, so the right metadata travels with every volume.

The data root defaults to the ``root`` entry in the catalog and can be
overridden with the ``DEEPVOID_DATA`` environment variable, which is how the
same catalog works on Colab, a cluster and a laptop.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .io import load_volume, read_fvol
from .volume import Volume

DEFAULT_CATALOG = Path(__file__).resolve().parent / "catalog.yaml"


class Catalog:
    """A set of named datasets read from a YAML file."""

    def __init__(self, path: str | os.PathLike = DEFAULT_CATALOG, root: str | None = None):
        with open(path) as f:
            spec = yaml.safe_load(f)
        self.root = Path(root or os.environ.get("DEEPVOID_DATA") or spec["root"])
        self.sims: dict[str, dict[str, Any]] = spec.get("simulations", {})
        self.entries: dict[str, dict[str, Any]] = spec["datasets"]

    def names(self, prefix: str = "") -> list[str]:
        """List dataset names, optionally only those starting with ``prefix``."""
        return sorted(n for n in self.entries if n.startswith(prefix))

    def path(self, name: str) -> Path:
        """Full path of a dataset's file."""
        return self.root / self._entry(name)["file"]

    def info(self, name: str) -> dict[str, Any]:
        """Metadata for a dataset: its own entry merged over its simulation's."""
        entry = self._entry(name)
        sim = dict(self.sims.get(entry["sim"], {}))
        sim.update({k: v for k, v in entry.items() if k not in ("file", "sim")})
        return sim

    def load(self, name: str) -> Volume:
        """Load a dataset as a :class:`Volume`."""
        path = self.path(name)
        info = self.info(name)
        if path.suffix == ".npz":
            vol = load_volume(path)
            vol.meta = {**info, **vol.meta}
            return vol
        data = read_fvol(path)
        if info.get("field") == "mask":
            data = data.astype(np.uint8)
        box_size = info.pop("box_size")
        return Volume(data=data, box_size=box_size, meta={"name": name, **info})

    def _entry(self, name: str) -> dict[str, Any]:
        try:
            return self.entries[name]
        except KeyError:
            raise KeyError(f"unknown dataset {name!r}; see Catalog.names()") from None
