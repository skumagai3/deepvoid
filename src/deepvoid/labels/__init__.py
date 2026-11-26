"""Ground-truth labels for the cosmic web.

Class ids are shared across the package:

    0 void, 1 wall, 2 filament, 3 halo
"""

VOID, WALL, FILAMENT, HALO = 0, 1, 2, 3
CLASS_NAMES = ("void", "wall", "filament", "halo")

from .tweb import smoothing_sigma, tweb_eigenvalues, tweb_labels  # noqa: E402
from .vweb import multiscale_correction, virial_overdensity, vweb_eigenvalues, vweb_labels  # noqa: E402

__all__ = ["VOID", "WALL", "FILAMENT", "HALO", "CLASS_NAMES",
           "smoothing_sigma", "tweb_eigenvalues", "tweb_labels",
           "vweb_eigenvalues", "vweb_labels", "multiscale_correction", "virial_overdensity"]
