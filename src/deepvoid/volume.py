"""The basic data object: a periodic 3D grid plus the metadata that describes it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Volume:
    """A cubic, periodic grid of one field (density, mask, tracer field, ...).

    Attributes
    ----------
    data:
        Array of shape (n, n, n). Axis order is (x, y, z).
    box_size:
        Side length of the box in h^-1 Mpc.
    meta:
        Free-form description of where the grid came from. Common keys:
        ``sim``, ``snapshot``, ``redshift``, ``field`` (``"density"``,
        ``"mask"``, ...), ``tracer`` (``"dm"``, ``"subhalos"``, ``"halos"``),
        ``spacing`` (mean tracer separation in h^-1 Mpc), ``rsd`` (bool),
        ``sigma_eff`` and ``threshold`` for masks.
    """

    data: np.ndarray
    box_size: float
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shape = self.data.shape
        if self.data.ndim != 3 or len(set(shape)) != 1:
            raise ValueError(f"expected a cubic 3D array, got shape {shape}")

    @property
    def n(self) -> int:
        """Number of cells per side."""
        return self.data.shape[0]

    @property
    def cell_size(self) -> float:
        """Cell size in h^-1 Mpc."""
        return self.box_size / self.n

    def __repr__(self) -> str:
        desc = ", ".join(f"{k}={v!r}" for k, v in self.meta.items())
        return (f"Volume(n={self.n}, box_size={self.box_size}, "
                f"dtype={self.data.dtype}, {desc})")
