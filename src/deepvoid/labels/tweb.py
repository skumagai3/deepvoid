"""Tidal-tensor (T-web) classification, following Hahn et al. (2007) with a
nonzero eigenvalue threshold (Forero-Romero et al. 2009).

The convention matches the DeepVoid dissertation (Sec. 5.3):

1. Density on an N^3 grid, as rho / rho_bar.
2. Gaussian smoothing with width sigma, chosen so that the effective
   smoothing sigma_eff^2 = sigma^2 + sigma_G^2 equals 1 h^-1 Mpc, where
   sigma_G = L_box / N is the grid size.
3. Tidal tensor T_ij = d_i d_j phi with the normalized Poisson equation
   del^2 phi = delta, computed in Fourier space as k_i k_j / k^2 delta_k.
4. Class = number of eigenvalues above lambda_th (0.65 by default):
   0 void, 1 wall, 2 filament, 3 halo.

Smoothing the density or the potential gives the same result, since both
are linear filters that commute.
"""

from __future__ import annotations

import numpy as np
import scipy.fft as sfft
from scipy import ndimage

from ..volume import Volume


def smoothing_sigma(box_size: float, n: int, sigma_eff: float = 1.0) -> float:
    """Gaussian sigma (in cells) that gives an effective smoothing ``sigma_eff``.

    Raises if the grid is too coarse to reach ``sigma_eff`` at all.
    """
    cell = box_size / n
    if cell >= sigma_eff:
        raise ValueError(f"cell size {cell:.3g} h^-1 Mpc is not below sigma_eff "
                         f"{sigma_eff} h^-1 Mpc; use a finer grid")
    return float(np.sqrt(sigma_eff ** 2 - cell ** 2) / cell)


def tweb_eigenvalues(density: np.ndarray, sigma_cells: float, slab: int = 32) -> np.ndarray:
    """Eigenvalues of the tidal tensor, ascending, shape (n, n, n, 3), float32.

    Memory peaks at about 13 times the size of the float32 grid
    (roughly 7 GB for 512^3).
    """
    d = (density / density.mean()).astype(np.float32)
    if sigma_cells > 0:
        d = ndimage.gaussian_filter(d, sigma_cells, mode="wrap")
    n = d.shape[0]

    d_k = sfft.rfftn(d, workers=-1)
    del d
    k = (sfft.fftfreq(n) * n).astype(np.float32)
    kx, ky = k[:, None, None], k[None, :, None]
    kz = (sfft.rfftfreq(n) * n).astype(np.float32)[None, None, :]
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k2[0, 0, 0] = 1.0  # the mean drops out: delta_k(0) * 0 / 1

    tensor = np.zeros((n, n, n, 3, 3), dtype=np.float32)
    pairs = {(0, 0): (kx, kx), (1, 0): (kx, ky), (1, 1): (ky, ky),
             (2, 0): (kx, kz), (2, 1): (ky, kz), (2, 2): (kz, kz)}
    for (i, j), (a, b) in pairs.items():
        tensor[..., i, j] = sfft.irfftn(d_k * (a * b / k2), s=(n, n, n), workers=-1)
    del d_k

    # eigvalsh reads only the lower triangle filled above.
    eig = np.empty((n, n, n, 3), dtype=np.float32)
    for s in range(0, n, slab):
        eig[s:s + slab] = np.linalg.eigvalsh(tensor[s:s + slab])
    return eig


def tweb_labels(density: Volume, sigma_eff: float = 1.0, threshold: float = 0.65,
                sigma_cells: float | None = None) -> Volume:
    """T-web class labels (uint8) for a density volume.

    ``sigma_cells`` overrides the smoothing computed from ``sigma_eff``, for
    reproducing legacy masks made with a fixed sigma in cells.
    """
    if sigma_cells is None:
        sigma_cells = smoothing_sigma(density.box_size, density.n, sigma_eff)
    eig = tweb_eigenvalues(density.data, sigma_cells)
    labels = (eig > threshold).sum(axis=-1).astype(np.uint8)
    meta = {k: v for k, v in density.meta.items() if k not in ("field", "name")}
    meta.update(field="mask", labels="tweb", threshold=threshold,
                sigma_cells=sigma_cells,
                sigma_eff=float(np.hypot(sigma_cells * density.cell_size, density.cell_size)))
    return Volume(data=labels, box_size=density.box_size, meta=meta)
