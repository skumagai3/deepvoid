"""Reading and writing volumes.

Two formats are supported:

``.fvol``
    The legacy binary format used throughout the original DeepVoid work: a
    256-byte header followed by float32 values in C order. Read it to get at
    existing files; there is no reason to write new ones except for
    compatibility with old scripts.

``.npz``
    The format used from now on. One compressed file per volume holding the
    array (stored as float16 for continuous fields and uint8 for masks) and a
    JSON copy of the volume's box size and metadata, so a file describes
    itself. A 512^3 density drops from 537 MB to roughly 150 to 250 MB, and a
    mask to a few tens of MB.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np

from .volume import Volume

FVOL_HEADER_BYTES = 256


def read_fvol(path: str | os.PathLike) -> np.ndarray:
    """Read a legacy ``.fvol`` file into a float32 array of shape (nx, ny, nz)."""
    with open(path, "rb") as f:
        header = f.read(FVOL_HEADER_BYTES)
        nx, ny, nz = struct.unpack("3i", header[12:24])
        data = np.fromfile(f, dtype=np.float32, count=nx * ny * nz)
    if data.size != nx * ny * nz:
        raise OSError(f"{path}: expected {nx * ny * nz} values, found {data.size}")
    return data.reshape(nx, ny, nz)


def write_fvol(array: np.ndarray, path: str | os.PathLike) -> None:
    """Write an array in the legacy ``.fvol`` format (stored as float32)."""
    nx, ny, nz = array.shape
    header = np.zeros(FVOL_HEADER_BYTES, dtype=np.uint8)
    header[:24] = np.frombuffer(
        np.array([8, 0, nx, nx, ny, nz], dtype="<i4").tobytes(), dtype=np.uint8)
    header[36:48] = np.frombuffer(np.ones(3, dtype="<f4").tobytes(), dtype=np.uint8)
    _atomic_write(path, lambda f: (header.tofile(f),
                                   np.ascontiguousarray(array, "<f4").tofile(f)))


def save_volume(volume: Volume, path: str | os.PathLike, dtype: str | None = None) -> None:
    """Save a volume as a compressed, self-describing ``.npz`` file.

    By default masks (integer data) are stored as uint8 and everything else as
    float16. Pass ``dtype`` to override, for example ``"float32"`` for a field
    whose dynamic range float16 cannot hold.
    """
    data = volume.data
    if dtype is None:
        dtype = "uint8" if np.issubdtype(data.dtype, np.integer) else "float16"
    if dtype == "float16":
        finite_max = np.nanmax(np.abs(data))
        if finite_max > np.finfo(np.float16).max:
            raise ValueError("values exceed the float16 range; normalize first or pass "
                             "dtype='float32'")
    header = json.dumps({"box_size": volume.box_size, "meta": volume.meta})
    _atomic_write(path, lambda f: np.savez_compressed(
        f, data=data.astype(dtype), header=np.array(header)))


def load_volume(path: str | os.PathLike, dtype: str | None = "float32") -> Volume:
    """Load a volume written by :func:`save_volume`.

    Continuous fields are returned as float32 by default; masks keep their
    integer type.
    """
    with np.load(path) as f:
        data = f["data"]
        header = json.loads(str(f["header"]))
    if dtype is not None and not np.issubdtype(data.dtype, np.integer):
        data = data.astype(dtype)
    return Volume(data=data, box_size=header["box_size"], meta=header["meta"])


def _atomic_write(path, write_fn) -> None:
    """Write to a temporary file, then rename, so a crash never leaves a partial file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write_fn(f)
    os.replace(tmp, path)
