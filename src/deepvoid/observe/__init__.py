"""Observational effects, applied to tracers before they are gridded.

Each function takes a :class:`~deepvoid.sources.Tracers` and returns a new
one, recording what it did in ``meta``. Effects compose in any order the
science calls for, for example::

    t = select_by_spacing(subhalos, spacing=7.0)
    t = to_redshift_space(t, axis=2)
    density = cic(t, n=512)

Labels are always computed from the true real-space matter field, so a model
trained on these inputs learns to recover the underlying structure.
"""

from .rsd import los_to_z, to_redshift_space
from .tracers import select_by_spacing

__all__ = ["los_to_z", "select_by_spacing", "to_redshift_space"]
