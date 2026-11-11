"""Saving a notebook's figures and numbers next to its results.

Colab notebooks with embedded plots get large and are awkward to share or
read back. A :class:`Report` writes each figure as a PNG and keeps the
numbers a notebook prints in ``results.json``, all in one folder, so the
results of a run can be looked at without opening the notebook.

    rep = Report(run_folder / "interpretability")
    rep.show(fig, "cka")                   # saves cka.png, then displays it
    rep.record(cka=curves)                 # merged into results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def plain(value):
    """Convert NumPy types and containers to JSON-friendly Python values."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class Report:
    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "results.json"
        self.results = json.loads(self.path.read_text()) if self.path.exists() else {}

    def show(self, fig, name: str, dpi: int = 110) -> Path:
        """Save ``fig`` as ``<name>.png`` in the report folder, then display it."""
        import matplotlib.pyplot as plt
        path = self.folder / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.show()
        return path

    def record(self, **values) -> None:
        """Add or replace entries in ``results.json``."""
        self.results.update(plain(values))
        self.path.write_text(json.dumps(self.results, indent=2))
