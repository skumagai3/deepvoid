"""Velocity-shear (V-web) classification, following Hoffman et al. (2012).

1. A smoothed peculiar-velocity field on an N^3 grid, in km/s
   (:func:`deepvoid.grid.velocity_field`; the smoothing is done there).
2. Normalized shear tensor
       Sigma_ij = -(d_i v_j + d_j v_i) / (2 H0),
   with H0 = 100 km/s per h^-1 Mpc, derivatives in Fourier space. With the
   minus sign a positive eigenvalue means compression along that axis.
3. Class = number of eigenvalues above the threshold:
   0 void, 1 wall (sheet), 2 filament, 3 halo (knot).

There is no standard threshold: Hoffman et al. chose 0.44 by eye for their
resolution and say the value has to be set for each problem, so
:func:`vweb_labels` requires one.

In linear theory v = -f H0 grad(phi) with del^2 phi = delta, which gives
Sigma = f T: the V-web and the T-web then agree, with thresholds in the ratio f.

Inside halos the velocity field is dominated by virial (random) motions, not
collapse, so on a fine grid some halo cores come out as void, wall or
filament. Hoffman et al. (2012, Sec. 5) correct this with a coarser V-web;
:func:`multiscale_correction` implements their rule. It is an optional step
on top of the V-web, and it changes labels only in overdense cells.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.fft as sfft

from ..volume import Volume
from . import FILAMENT, VOID, WALL
from .tensor import symmetric_eigenvalues

H0 = 100.0  # km/s per h^-1 Mpc


def _wavenumbers(n: int, box_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """kx, ky, kz in h Mpc^-1, shaped for broadcasting against an rfftn array.

    The Nyquist mode of an even grid is set to zero: a first derivative of
    that mode is not a real field, so it is dropped.
    """
    k = (2 * np.pi / box_size) * sfft.fftfreq(n) * n
    kz = (2 * np.pi / box_size) * sfft.rfftfreq(n) * n
    if n % 2 == 0:
        k[n // 2] = 0.0
        kz[-1] = 0.0
    k, kz = k.astype(np.float32), kz.astype(np.float32)
    return k[:, None, None], k[None, :, None], kz[None, None, :]


def vweb_eigenvalues(velocity: Sequence[np.ndarray], box_size: float,
                     slab: int = 32) -> np.ndarray:
    """Eigenvalues of the normalized shear tensor, ascending, shape (n, n, n, 3), float32.

    ``velocity`` is [v_x, v_y, v_z], each (n, n, n) in km/s; ``box_size`` in
    h^-1 Mpc. Memory peaks at about 16 times the size of one float32
    component, counting the input (roughly 9 GB at 512^3).
    """
    if len(velocity) != 3:
        raise ValueError("velocity must have three components")
    n = velocity[0].shape[0]
    kvec = _wavenumbers(n, box_size)
    v_k = [sfft.rfftn(np.asarray(v, dtype=np.float32), workers=-1) for v in velocity]

    comps = {}
    for i in range(3):
        for j in range(i + 1):
            # d_i v_j + d_j v_i  ->  i (k_i v_j + k_j v_i) in Fourier space
            s_k = 1j * (kvec[i] * v_k[j] + kvec[j] * v_k[i])
            comps[(i, j)] = (sfft.irfftn(s_k, s=(n, n, n), workers=-1, overwrite_x=True)
                             * np.float32(-0.5 / H0)).astype(np.float32, copy=False)
    del v_k
    return symmetric_eigenvalues(comps, slab)


def vweb_labels(velocity: Sequence[Volume], threshold: float) -> Volume:
    """V-web class labels (uint8) from the three velocity components.

    ``threshold`` has no default (see the module docstring).
    """
    box_size = velocity[0].box_size
    eig = vweb_eigenvalues([v.data for v in velocity], box_size)
    labels = (eig > threshold).sum(axis=-1).astype(np.uint8)
    meta = {k: v for k, v in velocity[0].meta.items()
            if k not in ("field", "name", "component", "units", "empty_cells")}
    meta.update(field="mask", labels="vweb", threshold=float(threshold))
    return Volume(data=labels, box_size=box_size, meta=meta)


def virial_overdensity(omega_m: float) -> float:
    """Virial density in units of the mean density, flat LCDM at z = 0.

    Bryan & Norman (1998): Delta_c = 18 pi^2 + 82 x - 39 x^2 relative to the
    critical density, x = Omega_m - 1; divided by Omega_m to refer it to the
    mean. About 331 for TNG (Omega_m = 0.3089). Hoffman et al. (2012) quote
    ~340 for their Omega_m = 0.28 without saying how it was computed; this
    formula gives 351 there.
    """
    x = omega_m - 1.0
    return float((18 * np.pi ** 2 + 82 * x - 39 * x ** 2) / omega_m)


def multiscale_correction(fine: Volume, coarse: Volume, density: np.ndarray,
                          virial: float) -> Volume:
    """Hoffman et al. (2012) multiscale correction of a V-web.

    Two kinds of fine-grid cells take the class of the coarse V-web:

    - voids with density at or above the mean (rho/rho_bar >= 1);
    - walls and filaments with density above the virial density
      (rho/rho_bar > ``virial``, e.g. :func:`virial_overdensity`).

    ``density`` is rho/rho_bar on the same grid, smoothed as the fine V-web.
    ``coarse`` is a V-web from a velocity field smoothed on a larger scale,
    same threshold. Every other cell keeps its fine class. The number of
    changed cells of each kind is recorded in the metadata.
    """
    rho = np.asarray(density)
    dense_void = (fine.data == VOID) & (rho >= 1.0)
    virial_sheet = ((fine.data == WALL) | (fine.data == FILAMENT)) & (rho > virial)
    take = dense_void | virial_sheet
    data = np.where(take, coarse.data, fine.data).astype(np.uint8)
    meta = dict(fine.meta)
    meta.update(multiscale="hoffman2012", coarse_sigma_eff=coarse.meta.get("sigma_eff"),
                virial=float(virial),
                corrected_dense_void=int(dense_void.sum()),
                corrected_virial=int(virial_sheet.sum()),
                changed=int((data != fine.data).sum()))
    return Volume(data=data, box_size=fine.box_size, meta=meta)
