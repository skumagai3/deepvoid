"""DeepVoid: semantic segmentation of the cosmic web with 3D U-Nets.

The NumPy parts (data, labels, splits, scoring, calibration) import without
TensorFlow; ``deepvoid.models``, ``deepvoid.losses``, ``deepvoid.pipeline``
and ``deepvoid.train`` need it.
"""

from .catalog import Catalog
from .io import load_volume, read_fvol, save_volume, write_fvol
from .project import Project
from .volume import Volume

__version__ = "0.1.0"
__all__ = ["Catalog", "Project", "Volume", "load_volume", "save_volume", "read_fvol", "write_fvol"]
