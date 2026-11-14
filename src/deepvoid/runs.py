"""A training run folder, for looking at results.

    from deepvoid.runs import Run

    run = Run("/content/drive/MyDrive/deepvoid_runs/subhalo_A_smooth2ch_L10")
    run.log["val_void_mcc"]          # one value per epoch
    run.scores()["tng300-3/subhalos/L10"]["void_f1"]
    scores = run.predict("tng300-3/subhalos/L10")   # (n, n, n, 4) class scores

Everything except :meth:`Run.predict` and :attr:`Run.model` works without
TensorFlow.
"""

from __future__ import annotations

import csv
import json
from functools import cached_property
from pathlib import Path

import numpy as np

from .catalog import Catalog
from .config import load_config
from .inputs import build
from .split import Region


class Run:
    def __init__(self, path: str | Path, catalog: Catalog | None = None):
        self.path = Path(path)
        if not (self.path / "config.yaml").exists():
            raise FileNotFoundError(f"{self.path} has no config.yaml; is it a run folder?")
        self._catalog = catalog

    @cached_property
    def catalog(self) -> Catalog:
        """The catalog the run's grids come from. Read on first use, so that
        listing run folders does not depend on every old config still loading."""
        cfg = dict(self.config)
        if cfg["catalog"] and not Path(cfg["catalog"]).exists():
            # A project folder that was moved: its catalog sits two levels up.
            cfg["catalog"] = str(self.path.parent.parent / "catalog.yaml")
        return Catalog.for_config(cfg, self._catalog)

    def __repr__(self) -> str:
        return f"Run({self.name!r})"

    @property
    def name(self) -> str:
        return self.path.name

    @cached_property
    def config(self) -> dict:
        return load_config(self.path / "config.yaml")

    @cached_property
    def regions(self) -> dict[str, Region]:
        """Train, validation and test slabs on the training grid."""
        raw = json.loads((self.path / "split.json").read_text())
        return {k: Region(**v) for k, v in raw.items()}

    @property
    def log(self) -> dict[str, np.ndarray]:
        """``log.csv`` as one array per column (re-read on every access)."""
        with open(self.path / "log.csv") as f:
            rows = list(csv.DictReader(f))
        return {k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])
                for k in rows[0]} if rows else {}

    @property
    def finished(self) -> bool:
        return (self.path / "test_scores.json").exists()

    def scores(self, prefer: str = "margin32") -> dict:
        """Test scores per input; ``test_scores_<prefer>.json`` if it exists."""
        for name in (f"test_scores_{prefer}.json", "test_scores.json"):
            if (self.path / name).exists():
                return json.loads((self.path / name).read_text())
        raise FileNotFoundError(f"{self.name} has no test scores yet")

    @cached_property
    def model(self):
        from .models import load_model
        return load_model(self.path / "best.keras")

    def input(self, name: str | None = None) -> np.ndarray:
        """A catalog field normalized the way this run saw its inputs: (n, n, n, channels)."""
        name = name or self.config["data"]["inputs"][0]
        norms = json.loads((self.path / "normalization.json").read_text())
        train = self.regions["train"].mask(self.catalog.load(self.config["data"]["labels"]).n)
        return build(self.catalog, name, self.config["data"]["smoothing"], norms, train,
                     self.config["data"]["normalization"])

    def labels(self) -> np.ndarray:
        """The T-web mask the run was trained on."""
        return self.catalog.load(self.config["data"]["labels"]).data

    def predict(self, name: str | None = None, **evaluate) -> np.ndarray:
        """Class scores (n, n, n, 4) for a whole box, as in evaluation.

        ``evaluate`` overrides the run's evaluate settings (crop, margin,
        batch_size, tta).
        """
        from .models import is_conditioned
        from .predict import predict_volume
        ev = {**self.config["evaluate"], **evaluate}
        name = name or self.config["data"]["inputs"][0]
        model = self.model
        if is_conditioned(model):
            spacing = float(self.catalog.info(name)["spacing"])

            def fn(b):
                s = np.full((len(b), 1, 1, 1, 1), spacing, np.float32)
                return model.predict_on_batch({"density": b, "spacing": s})
        else:
            fn = model.predict_on_batch
        return predict_volume(fn, self.input(name), crop=ev["crop"], margin=ev["margin"],
                              batch_size=ev["batch_size"], tta=ev["tta"])


def find_runs(root: str | Path, catalog: Catalog | None = None) -> list[Run]:
    """All run folders under ``root``, sorted by name."""
    return [Run(p.parent, catalog) for p in sorted(Path(root).glob("*/config.yaml"))]
