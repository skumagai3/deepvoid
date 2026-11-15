"""Trained DeepVoid models, ready to predict.

    from deepvoid import pretrained

    model = pretrained.load()                         # download once, then cached
    scores = model.predict(field, box_size=205.0, spacing=10.0)
    classes = scores.argmax(-1)                       # 0 void, 1 wall, 2 filament, 3 halo

A packaged model is a folder holding the network (``model.keras``) and
``model.json``, which records how its inputs were built. ``load`` downloads
one from Zenodo; :meth:`Pretrained.from_folder` opens a package on disk, and
:meth:`Pretrained.from_run` opens a training run folder directly.

What the model expects
----------------------
The input must be built the way the training inputs were:

- a periodic box gridded with cells of the size in ``model.json``
  (``cell_size``, in h^-1 Mpc), within ``cell_tolerance`` (5%);
- the mass of the most massive tracers assigned to the grid by
  cloud-in-cell, keeping enough of them that their mean separation is
  ``spacing`` (``deepvoid.Project.add_tracers`` does this);
- the whole box at once: the field is divided by its own mean.

Grids with a different cell size are refused rather than resampled, since
resampling changes the field the network sees.

Normalization
-------------
Each input channel is scaled with statistics saved at training time. By
default (``stats="saved"``) the statistics of the trained spacing nearest
to ``spacing`` are used, and spacings outside the trained range are refused.
This is how the model was tested. ``stats="refit"`` instead fits the
statistics on the box being predicted; it works at any spacing but has not
been tested the same way.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from .data import _download, _sha256
from .preprocess import Normalizer, normalizer

CELL_TOLERANCE = 0.05

# Published models. Each entry: Zenodo record, all-versions DOI, files.
MODELS: dict[str, dict] = {
    "tng300-3-subhalos": {
        "record": "24261833",
        "doi": "10.5281/zenodo.24261833",
        "description": "Trained on TNG300-3-Dark subhalos at mean separations "
                       "3, 5, 7 and 10 h^-1 Mpc at once (512^3 grid).",
    },
}
DEFAULT = "tng300-3-subhalos"
URL = "https://zenodo.org/records/{record}/files/{file}?download=1"


@dataclass
class Pretrained:
    """A trained network and the recipe for its inputs."""

    folder: Path
    info: dict
    _model: object = field(default=None, repr=False)

    @classmethod
    def from_folder(cls, folder: str | Path) -> "Pretrained":
        """A packaged model on disk (``model.keras`` and ``model.json``)."""
        folder = Path(folder)
        return cls(folder, json.loads((folder / "model.json").read_text()))

    @classmethod
    def from_run(cls, run_folder: str | Path, catalog=None) -> "Pretrained":
        """A training run folder, read as if it were a package."""
        from .runs import Run
        run = Run(run_folder, catalog)
        info = describe_run(run)
        info["weights"] = "best.keras"
        return cls(Path(run_folder), info)

    @property
    def model(self):
        if self._model is None:
            from .models import load_model
            self._model = load_model(self.folder / self.info.get("weights", "model.keras"))
        return self._model

    @property
    def spacings(self) -> list[float]:
        return sorted(float(s) for s in self.info["normalization"])

    def inputs(self, field: np.ndarray, box_size: float, spacing: float,
               stats: str = "saved") -> np.ndarray:
        """The normalized network input, shape (n, n, n, channels)."""
        field = np.asarray(field)
        if field.ndim != 3 or len(set(field.shape)) != 1:
            raise ValueError(f"field must be a cube (n, n, n); got shape {field.shape}")
        cell = box_size / field.shape[0]
        want = self.info["cell_size"]
        if abs(cell / want - 1) > CELL_TOLERANCE:
            n = round(box_size / want)
            raise ValueError(
                f"cell size {cell:.3f} h^-1 Mpc, but the model was trained on "
                f"{want:.3f}; grid this box on {n}^3 cells instead")
        if stats not in ("saved", "refit"):
            raise ValueError("stats must be 'saved' or 'refit'")
        if stats == "saved":
            lo, hi = self.spacings[0], self.spacings[-1]
            if not lo <= spacing <= hi:
                raise ValueError(
                    f"spacing {spacing} is outside the trained range {lo}-{hi} h^-1 Mpc; "
                    "use stats='refit' to fit the scaling on this box (untested)")
            nearest = min(self.spacings, key=lambda s: abs(s - spacing))
            norms = self.info["normalization"][_key(nearest)]
        channels = {"raw": field.astype(np.float32)}
        smoothing = self.info["smoothing"]
        if smoothing:
            sigma = smoothing["sigma_spacing"] * spacing / cell
            channels["smooth"] = ndimage.gaussian_filter(channels["raw"], sigma,
                                                         mode="wrap", truncate=4.0)
            if not smoothing.get("keep_raw", True):
                del channels["raw"]
        parts = []
        for name, data in channels.items():
            norm = (normalizer(norms[name]) if stats == "saved"
                    else Normalizer.fit(data, floor=self.info["floor"]))
            parts.append(norm(data))
        return np.stack(parts, axis=-1)

    def predict(self, field: np.ndarray, box_size: float, spacing: float,
                stats: str = "saved", tta: str = "none",
                batch_size: int | None = None) -> np.ndarray:
        """Class scores (n, n, n, 4) for a periodic box; ``argmax(-1)`` gives classes."""
        from .predict import predict_volume
        ev = self.info["evaluate"]
        x = self.inputs(field, box_size, spacing, stats)
        return predict_volume(self.model.predict_on_batch, x, crop=ev["crop"],
                              margin=ev["margin"], batch_size=batch_size or ev["batch_size"],
                              tta=tta)


def describe_run(run) -> dict:
    """``model.json`` contents for a training run: how its inputs were built."""
    cfg = run.config
    if cfg["model"].get("conditioning"):
        raise ValueError("models told the spacing (FiLM) are not packaged yet")
    if cfg["data"]["normalization"] != "log_standard":
        raise ValueError("only log_standard normalization is packaged")
    saved = json.loads((run.path / "normalization.json").read_text())
    normalization, cell_sizes = {}, set()
    for name in cfg["data"]["inputs"]:
        entry = {"raw": saved[name]} if name in saved else {}
        if f"{name}#smooth" in saved:
            entry["smooth"] = saved[f"{name}#smooth"]
        vol = run.catalog.load(name)          # read once for its cell size
        normalization[_key(vol.meta["spacing"])] = entry
        cell_sizes.add(round(vol.cell_size, 6))
        del vol
    if len(cell_sizes) != 1:
        raise ValueError(f"inputs have different cell sizes: {sorted(cell_sizes)}")
    floors = {v["floor"] for e in normalization.values() for v in e.values()}
    return {
        "name": run.name,
        "inputs": cfg["data"]["inputs"],
        "labels": cfg["data"]["labels"],
        "cell_size": cell_sizes.pop(),
        "smoothing": cfg["data"]["smoothing"],
        "floor": floors.pop(),
        "normalization": normalization,
        "evaluate": {k: cfg["evaluate"][k] for k in ("crop", "margin", "batch_size")},
        "classes": ["void", "wall", "filament", "halo"],
        "seed": cfg["seed"],
    }


def load(name: str = DEFAULT, folder: str | Path = "deepvoid_models",
         url: str = URL) -> Pretrained:
    """Download a published model into ``folder`` (once) and open it."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; available: {list(MODELS)}")
    entry = MODELS[name]
    if entry["record"] is None:
        raise RuntimeError(f"{name} is not published yet; open a run folder with "
                           "Pretrained.from_run(path) instead")
    target = Path(folder) / name
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "MANIFEST.txt"
    _download(url.format(record=entry["record"], file="MANIFEST.txt"), manifest_path)
    for line in manifest_path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        file, _size, digest, *_ = line.split()
        path = target / file
        if path.exists() and _sha256(path) == digest:
            continue
        print(f"downloading {file} ...", flush=True)
        _download(url.format(record=entry["record"], file=file), path)
        if _sha256(path) != digest:
            path.unlink()
            raise OSError(f"{file}: checksum does not match the record's MANIFEST.txt")
    return Pretrained.from_folder(target)


def _key(spacing: float) -> str:
    return f"{float(spacing):g}"
