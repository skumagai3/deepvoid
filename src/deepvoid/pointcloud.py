"""Tracer point clouds for 3D viewing in Potree (https://github.com/potree/potree).

Each tracer becomes a point carrying the class of the grid voxel it sits in,
written to a LAS 1.4 file that PotreeConverter turns into an octree for the
browser. The LAS ``classification`` field holds DeepVoid class ids
(0 void, 1 wall, 2 filament, 3 halo); other per-point values go into extra
attributes. Note that LAS reserves meanings for some classification codes
(2 is "ground" in lidar data); Potree only uses them for its default colors
and names, which :func:`write_viewer` replaces.

Grid convention: as in :mod:`deepvoid.grid`, grid point (i, j, k) sits at
position (i, j, k) * cell_size, so a tracer belongs to the voxel of its
nearest grid point, i = round(x / cell_size) mod n. Fields made with a
cell-centered convention, where voxel i spans [i, i + 1) * cell_size, need
``centered=True``; :func:`compare_with_field` tells the two apart.

Needs ``laspy`` to write LAS files (``pip install laspy``).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from .grid import cic
from .labels import CLASS_NAMES
from .sources import Tracers

# Okabe-Ito colors, readable on a dark background and for most color-blind viewers.
CLASS_COLORS = {0: (0.34, 0.71, 0.91), 1: (0.0, 0.62, 0.45),
                2: (0.90, 0.62, 0.0), 3: (0.80, 0.47, 0.65)}


def voxel_index(pos: np.ndarray, n: int, box_size: float, centered: bool = False) -> np.ndarray:
    """Index (N, 3) of the voxel each position falls in, on a periodic n^3 grid."""
    x = np.asarray(pos, dtype=np.float64) / (box_size / n)
    idx = np.floor(x) if centered else np.rint(x)
    return idx.astype(np.int64) % n


def values_at(data: np.ndarray, pos: np.ndarray, box_size: float,
              centered: bool = False) -> np.ndarray:
    """Values of a gridded field (n, n, n, ...) at the voxels holding ``pos``."""
    i, j, k = voxel_index(pos, data.shape[0], box_size, centered).T
    return data[i, j, k]


def compare_with_field(tracers: Tracers, field: np.ndarray) -> dict:
    """Check that ``tracers`` are the sample a gridded tracer field was made from.

    Grids the tracers with mass-weighted CIC on the field's grid, under both
    the integer-grid and the cell-centered convention, and returns the
    Pearson correlation of each with ``field``, plus counts of nonzero cells.
    The same sample gridded the same way gives a correlation of 1 up to
    float32 rounding; a different sample (another simulation or mass cut)
    gives a clearly lower value.
    """
    n = field.shape[0]
    f = np.asarray(field, dtype=np.float64).ravel()
    f = (f - f.mean()) / f.std()
    out = {"n_tracers": len(tracers), "nonzero_cells_field": int(np.count_nonzero(field))}
    half_cell = 0.5 * tracers.box_size / n
    for name, shift in (("integer", 0.0), ("centered", half_cell)):
        t = Tracers(pos=(tracers.pos - shift) % tracers.box_size, box_size=tracers.box_size,
                    mass=tracers.mass)
        g = cic(t, n).data.astype(np.float64).ravel()
        out[f"nonzero_cells_{name}"] = int(np.count_nonzero(g))
        g = (g - g.mean()) / g.std()
        out[f"correlation_{name}"] = float(np.mean(f * g))
    return out


def write_las(path: str | os.PathLike, pos: np.ndarray, classification: np.ndarray,
              extra: dict[str, np.ndarray] | None = None, scale: float = 1e-3) -> Path:
    """Write points to a LAS 1.4 file (point format 6).

    Parameters
    ----------
    pos:
        (N, 3) positions in h^-1 Mpc. LAS stores coordinates as integers
        times ``scale``; the default keeps them to 1 h^-1 kpc.
    classification:
        (N,) class ids, 0-255.
    extra:
        Further per-point values by name, stored as LAS extra attributes
        with their own dtype (for example ``{"log_mass": ...}``).
    """
    import laspy

    path = Path(path)
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([scale] * 3)
    header.offsets = np.zeros(3)
    extra = extra or {}
    header.add_extra_dims([laspy.ExtraBytesParams(name=k, type=np.asarray(v).dtype)
                           for k, v in extra.items()])
    las = laspy.LasData(header)
    las.x, las.y, las.z = np.asarray(pos, dtype=np.float64).T
    las.classification = np.asarray(classification, dtype=np.uint8)
    for k, v in extra.items():
        setattr(las, k, np.asarray(v))
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(path)
    return path


def classification_scheme(names=CLASS_NAMES, colors=CLASS_COLORS) -> dict:
    """Potree classification scheme: name and RGBA color per class id."""
    return {str(c): {"visible": True, "name": name, "color": [*colors[c], 1.0]}
            for c, name in enumerate(names)}


def write_viewer(template_page: str | os.PathLike, out: str | os.PathLike,
                 clouds: dict[str, str | os.PathLike], title: str,
                 point_size: float = 3.0, point_budget: int = 5_000_000) -> Path:
    """Write a Potree page showing several point clouds, colored by class.

    ``template_page`` is a page made by ``PotreeConverter --generate-page``;
    its library includes are kept and its viewer script is replaced. ``out``
    goes in the same folder, so the library paths stay valid. ``clouds`` maps
    a display name to the ``metadata.json`` of each converted cloud. Only the
    first cloud is visible at the start; the others are switched on in the
    sidebar's scene list.
    """
    template_page, out = Path(template_page), Path(out)
    html = template_page.read_text()
    entries = [{"name": name, "url": os.path.relpath(path, out.parent)}
               for name, path in clouds.items()]
    script = f"""<script>
    const viewer = new Potree.Viewer(document.getElementById("potree_render_area"));
    window.viewer = viewer;
    viewer.setEDLEnabled(true);
    viewer.setFOV(60);
    viewer.setPointBudget({point_budget});
    viewer.setBackground("black");
    if (viewer.setClassifications) {{
      viewer.setClassifications({json.dumps(classification_scheme())});
    }}
    viewer.loadGUI(() => {{
      viewer.setLanguage("en");
      $("#menu_scene").next().show();
      $("#menu_filters").next().show();
      viewer.toggleSidebar();
    }});
    const clouds = {json.dumps(entries)};
    clouds.forEach((c, i) => {{
      Potree.loadPointCloud(c.url, c.name, e => {{
        const pc = e.pointcloud;
        viewer.scene.addPointCloud(pc);
        pc.material.activeAttributeName = "classification";
        pc.material.pointSizeType = Potree.PointSizeType.FIXED;
        pc.material.size = {point_size};
        pc.visible = (i === 0);
        if (i === 0) viewer.fitToScreen();
      }});
    }});
  </script>"""
    blocks = list(re.finditer(r"<script\b[^>]*>(.*?)</script>", html, flags=re.S))
    viewer_blocks = [b for b in blocks if "Potree.Viewer" in b.group(1)]
    if not viewer_blocks:
        raise ValueError(f"{template_page} has no Potree viewer script to replace")
    b = viewer_blocks[-1]
    html = html[:b.start()] + script + html[b.end():]
    html = re.sub(r"<title>.*?</title>", f"<title>{title}</title>", html, flags=re.S)
    out.write_text(html)
    return out
