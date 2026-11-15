"""Package a training run as a model release (for Zenodo).

    python scripts/make_model_release.py RUN_FOLDER OUT_FOLDER

Writes to OUT_FOLDER:

- ``model.keras``: the run's ``best.keras``, unchanged;
- ``model.json``: how inputs must be built (cell size, smoothing, the
  normalization of each trained spacing, prediction crop and margin);
- ``config.yaml``, ``normalization.json``, ``test_scores.json`` and
  ``log.csv`` from the run, unchanged;
- ``MANIFEST.txt``: size and SHA-256 of every file, written last.

The packaged model is then opened once with ``Pretrained.from_folder`` and
must give the same class scores as the run folder on a test crop, so a
packaging mistake cannot be published.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from deepvoid.pretrained import Pretrained, describe_run
from deepvoid.runs import Run

COPIED = ["config.yaml", "normalization.json", "test_scores.json", "log.csv"]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main(run_folder: str, out: str) -> None:
    run = Run(run_folder)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run.path / "best.keras", out / "model.keras")
    for name in COPIED:
        shutil.copy2(run.path / name, out / name)
    info = describe_run(run)
    (out / "model.json").write_text(json.dumps(info, indent=2) + "\n")

    # The package must predict exactly what the run does.
    name = run.config["data"]["inputs"][-1]
    vol = run.catalog.load(name)
    packaged = Pretrained.from_folder(out)
    x_pkg = packaged.inputs(vol.data, vol.box_size, float(vol.meta["spacing"]))
    x_run = run.input(name)
    if not np.allclose(x_pkg, x_run, atol=1e-5):
        raise AssertionError("packaged inputs differ from the run's inputs")
    c = info["evaluate"]["crop"]
    crop = x_run[None, :c, :c, :c]
    if not np.allclose(packaged.model.predict_on_batch(crop),
                       run.model.predict_on_batch(crop), atol=1e-5):
        raise AssertionError("packaged model predicts differently from the run")
    print("package checked: same inputs and predictions as the run")

    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "MANIFEST.txt")
    with open(out / "MANIFEST.txt", "w") as f:
        f.write("# file  bytes  sha256\n")
        for p in files:
            f.write(f"{p.name}  {p.stat().st_size}  {file_sha256(p)}\n")
    print(f"wrote {len(files)} files and MANIFEST.txt to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
