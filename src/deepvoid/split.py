"""Spatial train / validation / test splits.

A box is cut into slabs along one axis before any patches are drawn, with
empty buffer slabs between the parts so that no training crop touches a
validation or test voxel. Because the box is periodic, the end of the test
slab is next to the start of the training slab, so a buffer goes there too.

    | train | buf | val | buf | test | buf |
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Region:
    """A slab of the box: cells ``start <= i < stop`` along ``axis``."""

    name: str
    axis: int
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start

    def mask(self, n: int) -> np.ndarray:
        """Boolean (n, n, n) array that is True inside the region."""
        m = np.zeros(n, dtype=bool)
        m[self.start:self.stop] = True
        shape = [1, 1, 1]
        shape[self.axis] = n
        return np.broadcast_to(m.reshape(shape), (n, n, n))

    def crop_origins(self, crop: int, n: int) -> tuple[range, range, range]:
        """Valid crop origins per axis so a crop of side ``crop`` stays inside."""
        full = range(0, n - crop + 1)
        along = range(self.start, self.stop - crop + 1)
        if len(along) == 0:
            raise ValueError(f"the {self.name!r} slab is {self.size} cells thick, thinner than "
                             f"the {crop}-cell crop; use a smaller crop or a larger slab")
        return tuple(along if a == self.axis else full for a in range(3))


def slab_split(n: int, fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
               buffer: int = 16, axis: int = 0, train_fraction: float = 1.0) -> dict[str, Region]:
    """Split an n-cell axis into train, val and test slabs with buffers.

    ``buffer`` (in cells) should be at least the receptive-field half-width
    you care about; with crops drawn entirely inside each slab, any buffer
    above zero already prevents shared voxels.

    ``train_fraction`` < 1 trains on only the first part of the training slab
    (rounded to whole cells) and leaves the rest unused; the validation and
    test slabs stay exactly where they are, so runs with less training data
    score on the same regions (data-limit tests).
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction}")
    usable = n - 3 * buffer
    sizes = [int(round(f * usable)) for f in fractions]
    sizes[0] = usable - sizes[1] - sizes[2]
    regions, start = {}, 0
    for name, size in zip(("train", "val", "test"), sizes):
        regions[name] = Region(name, axis, start, start + size)
        start += size + buffer
    if train_fraction < 1.0:
        t = regions["train"]
        regions["train"] = Region("train", axis, t.start, t.start + int(round(train_fraction * t.size)))
    return regions
