"""Simulation-specific readers.

Each module here knows one simulation: its cosmology, units and file layout,
and turns raw particles or catalogs into a :class:`deepvoid.sources.Tracers`
object in common units (positions in h^-1 Mpc, velocities in km/s, masses in
h^-1 Msun). Everything downstream of this package is simulation-agnostic.

Ready-made grids that already exist on disk are loaded through
:mod:`deepvoid.catalog` instead; these readers are for building new grids,
which is needed for redshift-space distortions, survey effects and new
tracer selections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Cosmology:
    """The few cosmological parameters the pipeline needs."""

    omega_m: float
    h: float

    def hubble_e(self, redshift: float) -> float:
        """E(z) = H(z)/H0 for a flat LCDM universe."""
        return float(np.sqrt(self.omega_m * (1 + redshift) ** 3 + 1 - self.omega_m))


@dataclass
class Tracers:
    """A set of point tracers (particles, halos, subhalos, galaxies) in one box.

    Attributes
    ----------
    pos:
        (N, 3) comoving positions in h^-1 Mpc, inside [0, box_size).
    box_size:
        Box side in h^-1 Mpc.
    mass:
        (N,) masses in h^-1 Msun, or None for equal-weight tracers.
    vel:
        (N, 3) peculiar velocities in km/s, or None if not available.
    meta:
        Description: ``sim``, ``redshift``, ``tracer``, ``cosmology`` ...
    """

    pos: np.ndarray
    box_size: float
    mass: np.ndarray | None = None
    vel: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.pos)

    def subset(self, index: np.ndarray) -> "Tracers":
        """Return the tracers selected by an index or boolean mask."""
        return Tracers(
            pos=self.pos[index],
            box_size=self.box_size,
            mass=None if self.mass is None else self.mass[index],
            vel=None if self.vel is None else self.vel[index],
            meta=dict(self.meta),
        )

    @property
    def spacing(self) -> float:
        """Mean tracer separation lambda = n^(-1/3) in h^-1 Mpc."""
        return self.box_size / len(self) ** (1 / 3)
