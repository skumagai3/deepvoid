"""Void catalogs from voxel classes: turning void voxels into separate voids.

The T-web void class usually percolates (one connected region spans the
box), so a catalog needs a rule for where one void ends and the next
begins. Each method here makes that rule an explicit parameter; none is
free of choices.

1. :func:`threshold_voids`: cores from a score threshold or an erosion of
   the void mask, labeled and grown back over the void voxels.
2. :func:`distance_voids`: basins of the distance to the nearest non-void
   voxel (a watershed), merged when a basin is a shallow dent in a bigger
   one (persistence) or when the neck between two is wide for their size.
3. :func:`soft_distance` + :func:`distance_voids`: the same, on a distance
   averaged over several void-score thresholds, so that walls the model is
   unsure of separate voids less than confident ones.
4. :func:`sphere_voids`: maximal empty spheres, after the idea of
   VoidFinder (El-Ad & Piran 1997; Hoyle & Vogeley 2002): the largest
   non-overlapping spheres are voids, smaller spheres that overlap one
   enough are added to it. The distance transform gives the largest empty
   sphere at every voxel, so no spheres are grown.

Everything assumes a periodic cube. Distances are in the units of ``cell``
(h^-1 Mpc if ``cell`` is the cell size). Positions follow the package grid
convention: point i at i * cell. NumPy and SciPy only
(no scikit-image: the basins are found by steepest ascent, which is exactly
periodic).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

__all__ = [
    "label_periodic", "distance_to_structure", "soft_distance", "local_peaks",
    "ascent_basins", "grow_labels", "merge_basins", "basin_persistence", "merge_tree", "cut_tree", "threshold_voids", "distance_voids",
    "sphere_voids", "void_catalog", "size_function", "match_voids", "percolation",
]


# ---------------------------------------------------------------------------
# Periodic building blocks
# ---------------------------------------------------------------------------

def _relabel(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Consecutive ids 1..n for the nonzero labels, 0 kept as background."""
    ids = np.unique(labels)
    ids = ids[ids > 0]
    lut = np.zeros(int(labels.max()) + 1 if labels.size else 1, dtype=np.int32)
    lut[ids] = np.arange(1, len(ids) + 1, dtype=np.int32)
    return lut[labels], len(ids)


def _union(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Component id of each node 0..n given edges a-b (node 0 stays alone)."""
    graph = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n + 1, n + 1))
    _, comp = connected_components(graph, directed=False)
    return comp


def label_periodic(mask: np.ndarray, connectivity: int = 1) -> tuple[np.ndarray, int]:
    """Connected components of a periodic cube.

    ``connectivity`` as in :func:`scipy.ndimage.generate_binary_structure`:
    1 = faces (6 neighbors), 2 = faces and edges (18), 3 = all 26.
    Components that touch across opposite faces get one label.
    """
    structure = ndimage.generate_binary_structure(3, connectivity)
    labels, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return labels.astype(np.int32), 0
    a_all, b_all = [], []
    for axis in range(3):
        first = np.take(labels, 0, axis=axis)
        last = np.take(labels, -1, axis=axis)
        # Neighbours across the face: an offset of +-1 along the face plane
        # is allowed when the full offset stays within the connectivity.
        for d1 in (-1, 0, 1):
            for d2 in (-1, 0, 1):
                if 1 + (d1 != 0) + (d2 != 0) > connectivity:
                    continue
                shifted = np.roll(last, (d1, d2), axis=(0, 1))
                both = (first > 0) & (shifted > 0)
                a_all.append(first[both])
                b_all.append(shifted[both])
    a = np.concatenate(a_all)
    b = np.concatenate(b_all)
    if len(a) == 0:
        return labels.astype(np.int32), n
    comp = _union(n, a, b)
    labels = np.where(mask, comp[labels] + 1, 0)   # +1: component 0 may be a real one
    return _relabel(labels)


def distance_to_structure(void: np.ndarray, cell: float = 1.0, margin: int = 64) -> np.ndarray:
    """Distance from each void voxel to the nearest non-void voxel (0 elsewhere).

    The radius of the largest sphere centered on the voxel that contains no
    non-void voxel center. Periodic: the box is padded by ``margin`` cells
    on every side, which is exact as long as every distance is below
    ``margin`` cells (checked).
    """
    void = np.asarray(void, dtype=bool)
    padded = np.pad(void, margin, mode="wrap")
    d = ndimage.distance_transform_edt(padded)
    d = d[margin:-margin, margin:-margin, margin:-margin].astype(np.float32)
    if d.max() >= margin:
        raise ValueError(f"a distance reaches {d.max():.0f} cells, not below the "
                         f"{margin}-cell padding; pass a larger margin")
    return d * np.float32(cell)


def soft_distance(score: np.ndarray, thresholds=(0.1, 0.3, 0.5, 0.7, 0.9),
                  cell: float = 1.0, margin: int = 64) -> np.ndarray:
    """Distance to structure averaged over void-score thresholds.

    For each threshold t the void region is ``score >= t``; the distances
    are averaged. A wall the model is sure of (low void score) separates
    voids at every threshold; one it is unsure of disappears at low
    thresholds, so the distance runs through it there.
    """
    total = np.zeros(score.shape, dtype=np.float32)
    for t in thresholds:
        total += distance_to_structure(score >= t, cell=cell, margin=margin)
    return total / len(thresholds)


def local_peaks(field: np.ndarray, mask: np.ndarray, min_value: float = 0.0) -> tuple[np.ndarray, int]:
    """Local maxima of ``field`` inside ``mask`` (26 neighbors, periodic),
    labeled; a flat top made of several voxels is one peak."""
    top = ndimage.maximum_filter(field, size=3, mode="wrap")
    peaks = (field >= top) & mask & (field >= min_value) & (field > 0)
    return label_periodic(peaks, connectivity=3)


_OFFSETS = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
            if (i, j, k) != (0, 0, 0)]


def ascent_basins(field: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Basins of steepest ascent on a periodic cube (a watershed, exactly
    periodic).

    Each voxel in ``mask`` points to its highest neighbor (26, periodic) if
    that neighbor is higher, and otherwise is a peak; following the
    pointers ends at a peak, and each peak's voxels form one basin. A flat
    top split between several peaks gives several basins whose pass equals
    their peak, so any merge rule in :func:`merge_basins` joins them again.
    """
    mask = np.asarray(mask, dtype=bool)
    f = np.where(mask, field, -np.inf).astype(np.float32)
    shape = f.shape
    best = f.copy()
    step = np.zeros(shape, dtype=np.int8)          # index into _OFFSETS + 1; 0 = peak
    for k, off in enumerate(_OFFSETS, start=1):
        g = np.roll(f, tuple(-o for o in off), axis=(0, 1, 2))   # g[x] = f[x + off]
        up = g > best
        best[up] = g[up]
        step[up] = k
        del g, up
    del best
    itype = np.int32 if f.size < 2 ** 31 else np.int64
    idx = np.arange(f.size, dtype=itype).reshape(shape)
    ptr = idx.copy()
    for k, off in enumerate(_OFFSETS, start=1):
        sel = step == k
        ptr[sel] = np.roll(idx, tuple(-o for o in off), axis=(0, 1, 2))[sel]
    del idx, step
    ptr = ptr.ravel()
    while True:                                     # pointer jumping
        nxt = ptr[ptr]
        if np.array_equal(nxt, ptr):
            break
        ptr = nxt
    roots = (ptr == np.arange(ptr.size, dtype=itype)) & mask.ravel()
    lab = np.zeros(ptr.size, dtype=np.int32)
    lab[roots] = np.arange(1, roots.sum() + 1, dtype=np.int32)
    labels = lab[ptr].reshape(shape)
    labels[~mask] = 0
    return labels, int(roots.sum())


def grow_labels(labels: np.ndarray, mask: np.ndarray, max_steps: int = 10_000) -> np.ndarray:
    """Grow labeled regions over the unlabeled voxels of ``mask``, one
    neighbor step (26, periodic) at a time, so each voxel joins the region
    it is connected to by the shortest path inside ``mask``. Where two
    fronts meet in the same step the higher id wins. Voxels connected to no
    region stay 0."""
    labels = labels.copy()
    for _ in range(max_steps):
        free = mask & (labels == 0)
        if not free.any():
            break
        grown = ndimage.maximum_filter(labels, size=3, mode="wrap")
        new = free & (grown > 0)
        if not new.any():
            break
        labels[new] = grown[new]
    return labels


def _pair_saddles(labels: np.ndarray, field: np.ndarray):
    """For each pair of touching labels: the highest value of
    min(field_i, field_j) over face-neighbor voxel pairs i, j on their
    shared boundary (the pass between them)."""
    n = int(labels.max())
    keys, vals = [], []
    for axis in range(3):
        other = np.roll(labels, -1, axis=axis)
        sel = (labels != other) & (labels > 0) & (other > 0)
        if not sel.any():
            continue
        a = labels[sel].astype(np.int64)
        b = other[sel].astype(np.int64)
        v = np.minimum(field[sel], np.roll(field, -1, axis=axis)[sel])
        keys.append(np.minimum(a, b) * (n + 1) + np.maximum(a, b))
        vals.append(v)
    if not keys:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    keys = np.concatenate(keys)
    vals = np.concatenate(vals)
    order = np.argsort(keys, kind="stable")
    keys, vals = keys[order], vals[order]
    starts = np.r_[0, np.flatnonzero(np.diff(keys)) + 1]
    saddle = np.maximum.reduceat(vals, starts)
    k = keys[starts]
    return k // (n + 1), k % (n + 1), saddle


def merge_basins(labels: np.ndarray, field: np.ndarray,
                 rule: Callable[[float, float, float], bool], return_pairs: bool = False):
    """Merge touching basins, highest pass first.

    For each touching pair, in order of decreasing pass value s (see
    :func:`_pair_saddles`), the two current groups are merged when
    ``rule(s, peak_a, peak_b)`` is true, where peak is the highest value of
    ``field`` in each group so far. This is the order ZOBOV merges zones in.

    Returns the merged labels, their number, and the merge history: one
    row (lower peak, pass) per merge, i.e. the birth and death of the basin
    that disappeared, for a persistence diagram. With ``return_pairs``,
    also, per merge, the labels (child, parent, child_edge, parent_edge):
    the peak basins of the lower and the higher group (elder rule), and the
    two touching basins whose pass joined them.
    """
    n = int(labels.max())
    if n == 0:
        out = (labels, 0, np.zeros((0, 2)))
        return out + (np.zeros((0, 4), np.int64),) if return_pairs else out
    peak = np.asarray(ndimage.maximum(field, labels, index=np.arange(n + 1)), dtype=np.float64)
    peak[0] = -np.inf
    a, b, s = _pair_saddles(labels, field)
    parent = np.arange(n + 1)
    history, pairs = [], []

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    for i in np.argsort(-s, kind="stable"):
        ra, rb = find(a[i]), find(b[i])
        if ra == rb:
            continue
        if rule(float(s[i]), peak[ra], peak[rb]):
            if peak[rb] > peak[ra]:
                ra, rb = rb, ra
            history.append((peak[rb], float(s[i])))
            ea, eb = (a[i], b[i]) if find(a[i]) == ra else (b[i], a[i])
            pairs.append((rb, ra, eb, ea))
            parent[rb] = ra
    roots = np.array([find(i) for i in range(n + 1)])
    merged, count = _relabel(roots[labels])
    out = (merged, count, np.array(history, dtype=np.float64).reshape(-1, 2))
    return out + (np.array(pairs, dtype=np.int64).reshape(-1, 4),) if return_pairs else out


def _drop_small(labels: np.ndarray, distance: np.ndarray, min_radius: float):
    """Remove voids whose largest empty-sphere radius is below ``min_radius``."""
    n = int(labels.max())
    if not n or min_radius <= 0:
        return labels
    r = np.asarray(ndimage.maximum(distance, labels, np.arange(1, n + 1)))
    small = np.flatnonzero(r < min_radius) + 1
    if len(small):
        labels = np.where(np.isin(labels, small), 0, labels)
        labels, _ = _relabel(labels)
    return labels


# ---------------------------------------------------------------------------
# The methods
# ---------------------------------------------------------------------------

def threshold_voids(void: np.ndarray, score: np.ndarray | None = None,
                    threshold: float | None = None, erode: int = 0,
                    min_core: int = 1) -> np.ndarray:
    """Option 1: separate cores, grown back over the void voxels.

    The cores are void voxels with ``score >= threshold`` (if given), eroded
    by ``erode`` cells (6-neighbor steps, periodic). Cores smaller than
    ``min_core`` voxels are dropped. Each remaining core is grown back over
    the void voxels connected to it (:func:`grow_labels`: each void voxel
    joins the core it is closest to along a path through void voxels).
    Void voxels connected to no core stay 0.
    """
    void = np.asarray(void, dtype=bool)
    core = void.copy()
    if threshold is not None:
        if score is None:
            raise ValueError("a threshold needs the void scores")
        core &= score >= threshold
    if erode:
        core = np.pad(core, erode + 1, mode="wrap")
        core = ndimage.binary_erosion(core, iterations=erode)
        core = core[erode + 1:-(erode + 1), erode + 1:-(erode + 1), erode + 1:-(erode + 1)]
    cores, n = label_periodic(core)
    if min_core > 1 and n:
        sizes = np.bincount(cores.ravel())
        small = np.flatnonzero(sizes < min_core)
        small = small[small > 0]
        if len(small):
            cores[np.isin(cores, small)] = 0
            cores, n = _relabel(cores)
    return grow_labels(cores, void)


def distance_voids(void: np.ndarray, distance: np.ndarray | None = None,
                   persistence: float | None = None, merge_ratio: float | None = None,
                   min_radius: float = 0.0, cell: float = 1.0, margin: int = 64,
                   return_history: bool = False):
    """Options 2 and 3: basins of the distance to structure, then merged.

    Basins grow from the local maxima of the distance (void centers; see
    :func:`ascent_basins`). Touching basins are merged, highest pass first,
    by one of two rules (give exactly one):

    - ``persistence``: merge while the smaller basin's peak is less than
      ``persistence`` (h^-1 Mpc) above the pass, i.e. it is a shallow dent
      in a bigger void. Small values keep every distinct dip; large values
      approach connected components.
    - ``merge_ratio``: merge while the pass is at least ``merge_ratio``
      times the smaller basin's peak (a neck wide for its size).

    Voids whose largest empty-sphere radius is below ``min_radius`` are
    dropped. Pass ``distance=soft_distance(...)`` for option 3. With
    ``return_history``, also returns the (birth, death) = (peak, pass) of
    every merged basin, for a persistence diagram.
    """
    if (persistence is None) == (merge_ratio is None):
        raise ValueError("give exactly one of persistence and merge_ratio")
    void = np.asarray(void, dtype=bool)
    if distance is None:
        distance = distance_to_structure(void, cell=cell, margin=margin)
    basins, _ = ascent_basins(distance, void)
    if persistence is not None:
        rule = lambda s, pa, pb: min(pa, pb) - s <= persistence  # noqa: E731
    else:
        rule = lambda s, pa, pb: s >= merge_ratio * min(pa, pb)  # noqa: E731
    labels, _, history = merge_basins(basins, distance, rule)
    labels = _drop_small(labels, distance, min_radius)
    return (labels, history) if return_history else labels


def basin_persistence(void: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """(birth, death) of every basin of the distance field: its peak and the
    pass at which it joins a higher one, merging everything that touches.
    Basins that never merge (one per connected void region) are left out.
    Points far below the diagonal are distinct voids; points near it are
    dents in a larger one."""
    basins, _ = ascent_basins(distance, void)
    _, _, history = merge_basins(basins, distance, lambda s, pa, pb: True)
    return history


def merge_tree(basins: np.ndarray, distance: np.ndarray) -> dict:
    """The full merge tree of the basins of the distance field.

    ``basins`` are the unmerged basins (:func:`ascent_basins`). Every
    touching pair is merged, highest pass first, as in
    :func:`basin_persistence`; each merge is one row: the ``child`` basin
    (peak basin of the lower group) joins the ``parent`` basin (peak basin of
    the higher group) at height ``pass_``. ``persistence = child_peak -
    pass_`` is the Δ above which the child is merged away. Per basin
    (index = label; index 0 unused): ``peak`` (its largest distance, the
    radius of its largest empty sphere), ``volume`` (in voxels) and
    ``peak_index`` (flat index of its peak voxel). :func:`cut_tree` turns
    the tree into the voids at any Δ: it needs ``child_edge`` and
    ``parent_edge``, the two touching basins whose pass is the merge (with
    a cut, the child joins whatever group the touching basin is in then,
    which need not contain ``parent``).
    """
    n = int(basins.max())
    _, _, hist, pairs = merge_basins(basins, distance, lambda s, pa, pb: True, return_pairs=True)
    idx = np.arange(n + 1)
    peak = np.asarray(ndimage.maximum(distance, basins, idx), dtype=np.float64)
    pos = ndimage.maximum_position(distance, basins, idx[1:])
    peak_index = np.zeros(n + 1, np.int64)
    if n:
        peak_index[1:] = np.ravel_multi_index(np.asarray(pos).T, basins.shape)
    peak[0] = 0.0
    return {"child": pairs[:, 0], "parent": pairs[:, 1], "child_edge": pairs[:, 2], "parent_edge": pairs[:, 3],
            "pass_": hist[:, 1], "child_peak": hist[:, 0],
            "persistence": hist[:, 0] - hist[:, 1], "peak": peak,
            "volume": np.bincount(basins.ravel(), minlength=n + 1).astype(np.int64), "peak_index": peak_index}


def cut_tree(basins: np.ndarray, tree: dict, persistence: float, distance: np.ndarray | None = None,
             min_radius: float = 0.0) -> np.ndarray:
    """Voids at Δ = ``persistence`` from a :func:`merge_tree`: every merge
    with ``child_peak - pass_ <= persistence`` is applied. Gives the same
    voids as :func:`distance_voids` with the same ``persistence`` (tested).
    ``min_radius`` needs ``distance``."""
    n = int(basins.max())
    parent = np.arange(n + 1)
    keep = np.asarray(tree["persistence"]) <= persistence

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for c, p in zip(np.asarray(tree["child_edge"])[keep], np.asarray(tree["parent_edge"])[keep]):
        rc, rp = find(c), find(p)
        if rc != rp:
            parent[rc] = rp
    roots = np.array([find(i) for i in range(n + 1)])
    labels, _ = _relabel(roots[basins])
    if min_radius > 0:
        if distance is None:
            raise ValueError("min_radius needs the distance field")
        labels = _drop_small(labels, distance, min_radius)
    return labels


def _overlap(r1, r2, d):
    """Volume shared by spheres of radii r1, r2 whose centers are d apart."""
    r1, r2, d = np.broadcast_arrays(np.asarray(r1, float), np.asarray(r2, float),
                                    np.asarray(d, float))
    out = np.zeros(d.shape)
    inside = d <= np.abs(r1 - r2)
    out[inside] = 4 / 3 * np.pi * np.minimum(r1, r2)[inside] ** 3
    part = (d < r1 + r2) & ~inside
    if part.any():
        a, b, c = r1[part], r2[part], d[part]
        out[part] = (np.pi * (a + b - c) ** 2
                     * (c ** 2 + 2 * c * (a + b) - 3 * (a - b) ** 2) / (12 * c))
    return out


def sphere_voids(distance: np.ndarray, void: np.ndarray, cell: float,
                 r_min: float = 10.0, r_hole: float | None = None,
                 max_overlap: float = 0.1, merge_overlap: float = 0.5) -> dict:
    """Option 4: voids as unions of empty spheres.

    Candidate spheres sit at the local maxima of ``distance`` (in h^-1 Mpc),
    with that distance as radius, so each contains no non-void voxel. In
    order of decreasing radius, a candidate of radius >= ``r_min`` is a
    maximal sphere (a new void) unless more than ``max_overlap`` of its
    volume lies in an earlier maximal sphere. Every other candidate of
    radius >= ``r_hole`` (default r_min / 2) joins the void whose maximal
    sphere covers the largest share of it, if that share is at least
    ``merge_overlap``. Defaults follow VoidFinder's published values (10
    h^-1 Mpc, 10%, 50%), which were chosen for galaxy surveys; they are a
    starting point here, not a calibration.

    Returns arrays ``center`` (N, 3), ``radius``, ``void`` (id, 1..),
    ``maximal`` (bool), and ``box_size``.
    """
    n = distance.shape[0]
    box = n * cell
    r_hole = r_min / 2 if r_hole is None else r_hole
    peaks, _ = local_peaks(distance, void, min_value=r_hole)
    idx = np.argwhere(peaks > 0)
    radius = distance[peaks > 0].astype(np.float64)
    empty = {"center": np.zeros((0, 3)), "radius": np.zeros(0), "void": np.zeros(0, np.int32),
             "maximal": np.zeros(0, bool), "box_size": box}
    if not len(radius):
        return empty
    order = np.argsort(-radius, kind="stable")
    idx, radius = idx[order], radius[order]
    center = (idx * cell) % box
    tree = cKDTree(center, boxsize=box)
    void_id = np.zeros(len(radius), dtype=np.int32)
    maximal = np.zeros(len(radius), dtype=bool)
    r_max = radius.max()
    vol = 4 / 3 * np.pi * radius ** 3

    def shared(i, js):
        sep = center[js] - center[i]
        sep -= box * np.round(sep / box)
        return _overlap(radius[i], radius[js], np.sqrt((sep ** 2).sum(1)))

    next_id = 1
    for i in np.flatnonzero(radius >= r_min):
        js = np.array(tree.query_ball_point(center[i], radius[i] + r_max), dtype=int)
        js = js[maximal[js]]
        if len(js) and shared(i, js).max() > max_overlap * vol[i]:
            continue
        maximal[i] = True
        void_id[i] = next_id
        next_id += 1
    for i in np.flatnonzero(~maximal):
        js = np.array(tree.query_ball_point(center[i], radius[i] + r_max), dtype=int)
        js = js[maximal[js]]
        if not len(js):
            continue
        ov = shared(i, js)
        k = int(np.argmax(ov))
        if ov[k] >= merge_overlap * vol[i]:
            void_id[i] = void_id[js[k]]
    keep = void_id > 0
    return {"center": center[keep], "radius": radius[keep], "void": void_id[keep],
            "maximal": maximal[keep], "box_size": box}


def sphere_union_volume(spheres: dict, samples: int = 2000, seed: int = 0) -> np.ndarray:
    """Volume of each sphere void (union of its spheres), by Monte Carlo.

    Each sphere gets ``samples`` uniform points; a point counts 1/k when it
    lies in k spheres of its void, which makes the sum an unbiased estimate
    of the union's volume.
    """
    rng = np.random.default_rng(seed)
    box = spheres["box_size"]
    ids = spheres["void"]
    out = np.zeros(ids.max() + 1 if len(ids) else 1)
    for v in np.unique(ids):
        sel = np.flatnonzero(ids == v)
        c, r = spheres["center"][sel], spheres["radius"][sel]
        for k in range(len(sel)):
            u = rng.normal(size=(samples, 3))
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            pts = c[k] + u * r[k] * rng.random(samples)[:, None] ** (1 / 3)
            sep = pts[:, None, :] - c[None, :, :]
            sep -= box * np.round(sep / box)
            count = ((sep ** 2).sum(-1) < r[None, :] ** 2).sum(1)
            out[v] += 4 / 3 * np.pi * r[k] ** 3 * np.mean(1.0 / np.maximum(count, 1))
    return out[1:]


# ---------------------------------------------------------------------------
# Catalogs, size functions, matching
# ---------------------------------------------------------------------------

def void_catalog(labels: np.ndarray, cell: float, distance: np.ndarray | None = None,
                 slab: int = 32) -> dict:
    """Per-void volume, effective radius and center for labeled voids.

    ``center`` is the volume-weighted center, averaged on the circle along
    each axis so that voids crossing a face come out right. With
    ``distance``, also ``r_max`` (largest empty-sphere radius) and
    ``deepest`` (where it sits), an alternative center.
    """
    n_side = labels.shape[0]
    box = n_side * cell
    n = int(labels.max())
    count = np.bincount(labels.ravel(), minlength=n + 1).astype(np.float64)
    trig = np.zeros((3, 2, n + 1))
    phase = 2 * np.pi * np.arange(n_side) / n_side
    for x0 in range(0, n_side, slab):
        lab = labels[x0:x0 + slab]
        flat = lab.ravel()
        grids = np.meshgrid(phase[x0:x0 + lab.shape[0]], phase, phase, indexing="ij")
        for axis in range(3):
            g = grids[axis].ravel()
            trig[axis, 0] += np.bincount(flat, weights=np.cos(g), minlength=n + 1)
            trig[axis, 1] += np.bincount(flat, weights=np.sin(g), minlength=n + 1)
    angle = np.arctan2(trig[:, 1], trig[:, 0]) % (2 * np.pi)
    center = (angle / (2 * np.pi) * box).T[1:]
    volume = count[1:] * cell ** 3
    out = {"id": np.arange(1, n + 1), "volume": volume,
           "r_eff": (3 * volume / (4 * np.pi)) ** (1 / 3), "center": center,
           "box_size": box}
    if distance is not None and n:
        idx = np.arange(1, n + 1)
        out["r_max"] = np.asarray(ndimage.maximum(distance, labels, idx), dtype=np.float64)
        pos = ndimage.maximum_position(distance, labels, idx)
        out["deepest"] = np.asarray(pos, dtype=np.float64) * cell
    return out


def size_function(radius: np.ndarray, box_size: float, bins: np.ndarray) -> np.ndarray:
    """Cumulative number density n(>R) in (h^-1 Mpc)^-3 at each bin edge."""
    radius = np.sort(np.asarray(radius))
    above = len(radius) - np.searchsorted(radius, bins, side="left")
    return above / box_size ** 3


def match_voids(labels_a: np.ndarray, labels_b: np.ndarray) -> dict:
    """For each void in ``a``, the void in ``b`` sharing most voxels.

    Returns ``best`` (id in b, 0 if none), ``shared`` voxels, and
    ``jaccard`` = shared / (|a| + |b| - shared).
    """
    a = labels_a.ravel()
    b = labels_b.ravel()
    na, nb = int(a.max()), int(b.max())
    size_a = np.bincount(a, minlength=na + 1)
    size_b = np.bincount(b, minlength=nb + 1)
    both = (a > 0) & (b > 0)
    key = a[both].astype(np.int64) * (nb + 1) + b[both]
    key, cnt = np.unique(key, return_counts=True)
    ka, kb = key // (nb + 1), key % (nb + 1)
    best = np.zeros(na + 1, dtype=np.int64)
    shared = np.zeros(na + 1, dtype=np.int64)
    order = np.lexsort((cnt, ka))           # by a, then by count
    last = np.r_[np.flatnonzero(np.diff(ka[order])), len(order) - 1]
    pick = order[last]
    best[ka[pick]] = kb[pick]
    shared[ka[pick]] = cnt[pick]
    union = size_a + size_b[best] - shared
    with np.errstate(invalid="ignore", divide="ignore"):
        jac = np.where(best > 0, shared / union, 0.0)
    return {"best": best[1:], "shared": shared[1:], "jaccard": jac[1:]}


def percolation(void: np.ndarray) -> dict:
    """How connected the void region is: number of periodic components and
    the share of void voxels in the largest."""
    labels, n = label_periodic(void)
    if n == 0:
        return {"components": 0, "largest_share": 0.0}
    sizes = np.bincount(labels.ravel())[1:]
    return {"components": n, "largest_share": float(sizes.max() / sizes.sum())}
