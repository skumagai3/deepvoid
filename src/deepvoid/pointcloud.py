"""Tracer point clouds for 3D viewing in Potree (https://github.com/potree/potree).

Each tracer becomes a point carrying the class of the grid voxel it sits in,
written to a LAS 1.4 file that PotreeConverter turns into an octree for the
browser. The LAS ``classification`` field holds DeepVoid class ids
(0 void, 1 wall, 2 filament, 3 halo); other per-point values go into extra
attributes. Voxel clouds (model errors, void surfaces, density) use further
codes, listed in ``POINT_CODES``, so one color scheme serves a whole page.
Note that LAS reserves meanings for some classification codes
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

# Point codes beyond the four classes, for voxel clouds. One scheme covers
# every cloud on a page, so each kind of point gets its own code.
WALL_AS_VOID, VOID_AS_WALL, SURFACE_TRUE, SURFACE_PRED, DENSITY = 4, 5, 6, 7, 8
POINT_CODES = {
    **{c: (name, CLASS_COLORS[c]) for c, name in enumerate(CLASS_NAMES)},
    WALL_AS_VOID: ("true wall, predicted void", (0.84, 0.37, 0.0)),
    VOID_AS_WALL: ("true void, predicted wall", (0.94, 0.89, 0.26)),
    SURFACE_TRUE: ("void surface, T-web", (0.34, 0.71, 0.91)),
    SURFACE_PRED: ("void surface, predicted", (0.0, 0.45, 0.70)),
    DENSITY: ("dark matter above density cut", (0.6, 0.6, 0.6)),
}


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


def voxel_positions(selected: np.ndarray, box_size: float) -> np.ndarray:
    """Positions (N, 3) in h^-1 Mpc of the grid points where ``selected`` is True."""
    n = selected.shape[0]
    return np.argwhere(selected).astype(np.float64) * (box_size / n)


def void_surface(classes: np.ndarray) -> np.ndarray:
    """Void voxels with at least one non-void face neighbor, on a periodic grid.

    These form a shell one voxel thick around each void, a small fraction
    of the void volume, which shows void shapes when drawn as points.
    """
    void = classes == 0
    edge = np.zeros_like(void)
    for axis in range(3):
        for step in (1, -1):
            edge |= ~np.roll(void, step, axis=axis)
    return void & edge


def slab(n: int, axis: int, start: int, stop: int) -> np.ndarray:
    """Boolean (n, n, n) array, True for cells start <= i < stop along ``axis``."""
    from .split import Region
    return Region("slab", axis, start, stop).mask(n)


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


def classification_scheme(codes: dict = POINT_CODES) -> dict:
    """Potree classification scheme: name and RGBA color per point code."""
    return {str(c): {"visible": True, "name": name, "color": [*color, 1.0]}
            for c, (name, color) in codes.items()}


def write_viewer(template_page: str | os.PathLike, out: str | os.PathLike,
                 clouds: list[dict], title: str, point_budget: int = 5_000_000) -> Path:
    """Write a Potree page showing several point clouds.

    ``template_page`` is a page made by ``PotreeConverter --generate-page``;
    its library includes are kept and its viewer script is replaced. ``out``
    goes in the same folder, so the library paths stay valid.

    Each entry of ``clouds`` describes one cloud:

    - ``name``: shown in the sidebar's scene list
    - ``metadata``: path of the converted cloud's ``metadata.json``
    - ``attribute``: what colors the points, ``"classification"`` (default)
      or an extra attribute such as ``"log_density"``
    - ``gradient``: Potree color scale for a numeric attribute (default
      ``"INFERNO"``), and ``range``: its [min, max]
    - ``size``: point size in pixels (default 3)
    - ``visible``: shown at start-up (default True for the first cloud only)
    """
    template_page, out = Path(template_page), Path(out)
    html = template_page.read_text()
    entries = []
    for i, c in enumerate(clouds):
        entries.append({"name": c["name"],
                        "url": os.path.relpath(c["metadata"], out.parent),
                        "attribute": c.get("attribute", "classification"),
                        "gradient": c.get("gradient", "INFERNO"),
                        "range": c.get("range"),
                        "size": c.get("size", 3),
                        "visible": c.get("visible", i == 0)})
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
    let fitted = false;
    clouds.forEach(c => {{
      Potree.loadPointCloud(c.url, c.name, e => {{
        const pc = e.pointcloud;
        viewer.scene.addPointCloud(pc);
        pc.material.activeAttributeName = c.attribute;
        if (c.attribute !== "classification") {{
          pc.material.gradient = Potree.Gradients[c.gradient];
          if (c.range) pc.material.setRange(c.attribute, c.range);
        }}
        pc.material.pointSizeType = Potree.PointSizeType.FIXED;
        pc.material.size = c.size;
        pc.visible = c.visible;
        if (c.visible && !fitted) {{ viewer.fitToScreen(); fitted = true; }}
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


SERVE_SCRIPT = '''"""Serve this folder for the Potree pages: python3 serve.py [port]

Potree requests many parts of a point cloud at once; Python's default
server queues only five connections and refuses the rest, which leaves
parts of a cloud unloaded. This one queues up to 256.
"""
import http.server
import sys


class Server(http.server.ThreadingHTTPServer):
    request_queue_size = 256
    daemon_threads = True


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
print(f"open http://localhost:{port}/index.html  (Ctrl-C to stop)")
Server(("localhost", port), http.server.SimpleHTTPRequestHandler).serve_forever()
'''


def build_page(converter: str | os.PathLike, page_dir: str | os.PathLike,
               clouds: list[dict], title: str, page_name: str = "index") -> Path:
    """Convert LAS files with PotreeConverter and write a viewer page for them.

    ``clouds`` is as in :func:`write_viewer`, with ``las`` (the LAS file)
    in place of ``metadata``. Converted clouds go to
    ``page_dir/pointclouds/<LAS file name>``; the Potree libraries are
    copied into ``page_dir`` from a page PotreeConverter generates, along
    with ``serve.py``, a local web server for viewing the pages. Several
    pages can share one ``page_dir``.
    """
    import shutil
    import subprocess
    import tempfile

    page_dir = Path(page_dir)
    page_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([str(converter), str(clouds[0]["las"]), "-o", tmp,
                        "--generate-page", "template"], check=True, capture_output=True)
        template = next(p for p in Path(tmp).rglob("*.html") if "Potree.Viewer" in p.read_text())
        shutil.copytree(template.parent / "libs", page_dir / "libs", dirs_exist_ok=True)
        shutil.copy(template, page_dir / "template.html")
    (page_dir / "serve.py").write_text(SERVE_SCRIPT)

    entries = []
    for c in clouds:
        target = page_dir / "pointclouds" / Path(c["las"]).stem
        shutil.rmtree(target, ignore_errors=True)
        subprocess.run([str(converter), str(c["las"]), "-o", str(target)],
                       check=True, capture_output=True)
        entries.append({**{k: v for k, v in c.items() if k != "las"},
                        "metadata": target / "metadata.json"})
    out = write_viewer(page_dir / "template.html", page_dir / f"{page_name}.html", entries, title)
    (page_dir / "template.html").unlink()
    return out
