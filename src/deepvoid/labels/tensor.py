"""Eigenvalues of a symmetric 3x3 tensor field, shared by the T-web and V-web."""

from __future__ import annotations

import numpy as np


def symmetric_eigenvalues(components: dict[tuple[int, int], np.ndarray],
                          slab: int = 32) -> np.ndarray:
    """Eigenvalues, ascending, of a symmetric tensor field given by its lower triangle.

    ``components`` maps (i, j) with i >= j to an (n, n, n) array; all six
    entries (0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2) are needed.
    Returns shape (n, n, n, 3), float32. The tensor is assembled one slab of
    ``slab`` x-planes at a time to keep memory low; ``eigvalsh`` reads only
    the lower triangle.
    """
    n = components[(0, 0)].shape[0]
    eig = np.empty((n, n, n, 3), dtype=np.float32)
    for s in range(0, n, slab):
        t = np.zeros((min(slab, n - s), n, n, 3, 3), dtype=np.float32)
        for (i, j), c in components.items():
            t[..., i, j] = c[s:s + slab]
        eig[s:s + slab] = np.linalg.eigvalsh(t)
    return eig
