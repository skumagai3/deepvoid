"""Turning catalog volumes into network inputs.

An input is one or more channels built from a single density field:

- the field itself (the raw channel);
- optionally, the same field Gaussian-smoothed (the smoothed channel),
  set by ``data.smoothing`` in the config.

At low tracer density a gridded tracer field is almost all empty cells (at
spacing 10 h^-1 Mpc on a 512^3 grid, 99.95% of them), so the first layers of
the network see little but zeros. The smoothed channel fills in the space
between tracers; keeping the raw channel beside it means nothing is lost,
and the network can weigh the two.

Each channel is normalized separately (``preprocess.Normalizer``), with
statistics from the training slab only. Smoothing is applied to the whole
periodic box, so near a slab edge it mixes in tracers from the next slab.
Only inputs mix this way, never labels, and at test time the network sees
the whole box anyway.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .catalog import Catalog
from .preprocess import MinMax, Normalizer, normalizer


def smoothing_cells(smoothing: dict, spacing: float | None, cell_size: float) -> float:
    """Gaussian sigma in cells from ``{"sigma_mpc": s}`` or ``{"sigma_spacing": f}``."""
    if "sigma_mpc" in smoothing:
        sigma = float(smoothing["sigma_mpc"])
    elif "sigma_spacing" in smoothing:
        if spacing is None:
            raise ValueError("sigma_spacing needs the input's 'spacing' in the catalog")
        sigma = float(smoothing["sigma_spacing"]) * spacing
    else:
        raise ValueError("data.smoothing needs 'sigma_mpc' or 'sigma_spacing'")
    return sigma / cell_size


def channels(catalog: Catalog, name: str, smoothing: dict | None) -> dict[str, np.ndarray]:
    """The un-normalized channels for one input, keyed by channel name."""
    vol = catalog.load(name)
    out = {name: vol.data}
    if smoothing:
        unknown = set(smoothing) - {"sigma_mpc", "sigma_spacing", "keep_raw"}
        if unknown:
            raise KeyError(f"unknown data.smoothing settings {sorted(unknown)}")
        sigma = smoothing_cells(smoothing, vol.meta.get("spacing"), vol.cell_size)
        out[f"{name}#smooth"] = ndimage.gaussian_filter(
            vol.data.astype(np.float32), sigma, mode="wrap", truncate=4.0)
        if not smoothing.get("keep_raw", True):
            del out[name]
    return out


def build(catalog: Catalog, name: str, smoothing: dict | None, norms: dict,
          train_mask: np.ndarray | None = None, kind: str = "log_standard") -> np.ndarray:
    """Normalized input of shape (n, n, n, channels).

    ``norms`` maps channel names to saved :class:`Normalizer` settings (as
    dicts); a channel missing from it is fit on ``train_mask`` and added.
    ``kind`` chooses what is fit: ``"log_standard"`` (:class:`Normalizer`) or
    ``"minmax"`` (:class:`MinMax`, the first paper's scaling).
    """
    if kind not in ("log_standard", "minmax"):
        raise ValueError("data.normalization must be 'log_standard' or 'minmax'")
    parts = []
    for key, data in channels(catalog, name, smoothing).items():
        if key not in norms:
            if train_mask is None:
                raise ValueError(f"no saved normalization for {key!r}")
            fit = MinMax.fit if kind == "minmax" else Normalizer.fit
            norms[key] = fit(data, train_mask).to_dict()
        parts.append(normalizer(norms[key])(data))
    return np.stack(parts, axis=-1)


def n_channels(smoothing: dict | None) -> int:
    if not smoothing:
        return 1
    return 2 if smoothing.get("keep_raw", True) else 1
