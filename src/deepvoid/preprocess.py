"""Input normalization and class weights.

Normalization statistics are always fit on the training region only and saved with the
model, so validation, test and new volumes are scaled exactly as the
training data were.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

FLOOR = 1e-3  # rho / rho_bar assigned to empty voxels before the log


@dataclass
class Normalizer:
    """log10(rho / rho_bar + floor), then standardized: x -> (x - mean) / std.

    Apply it to whole volumes, not crops: rho_bar is the mean of the array
    passed in, which is the box mean only for a full box.
    """

    mean: float
    std: float
    floor: float = FLOOR

    @staticmethod
    def log_density(density: np.ndarray, floor: float = FLOOR) -> np.ndarray:
        return np.log10(density / density.mean() + floor).astype(np.float32)

    @classmethod
    def fit(cls, density: np.ndarray, region_mask: np.ndarray | None = None,
            floor: float = FLOOR) -> "Normalizer":
        x = cls.log_density(density, floor)
        sample = x if region_mask is None else x[region_mask]
        return cls(mean=float(sample.mean()), std=float(sample.std()), floor=floor)

    def __call__(self, density: np.ndarray) -> np.ndarray:
        return (self.log_density(density, self.floor) - self.mean) / self.std

    def to_dict(self) -> dict:
        return asdict(self)


def auto_class_weights(labels: np.ndarray, n_classes: int = 4) -> list[float]:
    """Class weights from label frequencies: w_c proportional to f_c^(-1/2).

    The square root keeps very rare classes (halos, ~0.5% of voxels) from
    dominating. Weights are scaled so the average weight per voxel is 1,
    which keeps the loss on the same scale as unweighted cross-entropy.
    """
    f = np.bincount(labels.ravel(), minlength=n_classes) / labels.size
    w = np.where(f > 0, f, 1.0) ** -0.5
    w = w / np.sum(f * w)
    return [float(x) for x in w]
