"""Tests for deepvoid.voids on small synthetic periodic boxes."""

import numpy as np
import pytest

from deepvoid import voids as V


def ball(n, center, radius):
    """Boolean ball on a periodic n^3 grid (center in cells)."""
    g = np.arange(n)
    d2 = 0
    for c, ax in zip(center, np.meshgrid(g, g, g, indexing='ij')):
        sep = ax - c
        sep = sep - n * np.round(sep / n)
        d2 = d2 + sep ** 2
    return d2 <= radius ** 2


def test_label_periodic_joins_across_faces():
    n = 24
    mask = ball(n, (0, 12, 12), 4)          # cut in two by the x faces
    mask |= ball(n, (12, 12, 12), 3)        # a separate ball
    labels, count = V.label_periodic(mask)
    assert count == 2
    assert labels[0, 12, 12] == labels[n - 1, 12, 12]
    assert labels[0, 12, 12] != labels[12, 12, 12]


def test_label_periodic_edge_connectivity():
    n = 10
    mask = np.zeros((n, n, n), bool)
    mask[0, 0, 5] = True
    mask[n - 1, 1, 5] = True                # touches the first only along an edge
    assert V.label_periodic(mask, connectivity=1)[1] == 2
    assert V.label_periodic(mask, connectivity=2)[1] == 1


def test_distance_is_periodic_and_exact():
    n = 20
    void = np.ones((n, n, n), bool)
    void[5, :, :] = False                   # one wall at x = 5
    d = V.distance_to_structure(void, cell=2.0, margin=12)
    # Nearest wall copy is at x = 5 or x = 25: distance min(|x-5|, |x-25|) cells.
    x = np.arange(n)
    want = np.minimum(np.abs(x - 5), np.abs(x - 25)) * 2.0
    assert np.allclose(d[:, 3, 7], want)
    with pytest.raises(ValueError):
        V.distance_to_structure(void, margin=5)


def dumbbell(n=48):
    """Two empty balls (radii 9 and 7 cells) joined by a 2-cell neck."""
    void = ball(n, (14, 24, 24), 9) | ball(n, (34, 24, 24), 7)
    g = np.arange(n)
    x, y, z = np.meshgrid(g, g, g, indexing='ij')
    void |= (x >= 14) & (x <= 34) & ((y - 24) ** 2 + (z - 24) ** 2 <= 4)
    return void


def test_distance_voids_split_and_merge():
    void = dumbbell()
    # Smaller ball's peak ~7 cells, neck pass ~2: depth ~5, ratio ~0.3.
    for kw_split, kw_join in (({'persistence': 3.0}, {'persistence': 6.0}),
                              ({'merge_ratio': 0.5}, {'merge_ratio': 0.1})):
        split = V.distance_voids(void, margin=16, **kw_split)
        joined = V.distance_voids(void, margin=16, **kw_join)
        assert split.max() == 2
        assert joined.max() == 1
        assert split[14, 24, 24] != split[34, 24, 24]
        assert (split > 0).sum() == void.sum()   # every void voxel assigned
    with pytest.raises(ValueError):
        V.distance_voids(void, margin=16)


def test_distance_voids_across_the_faces():
    # The same dumbbell shifted so that both balls and the neck wrap around.
    void = np.roll(dumbbell(), (24, 20, 30), axis=(0, 1, 2))
    split = V.distance_voids(void, persistence=3.0, margin=16)
    assert split.max() == 2
    labels, hist = V.distance_voids(void, persistence=6.0, margin=16, return_history=True)
    assert labels.max() == 1
    # The most persistent merge is the smaller ball joining the larger one;
    # the rest are dents in the neck or flat tops.
    birth, death = hist[np.argmax(hist[:, 0] - hist[:, 1])]
    assert 6.5 <= birth <= 7.5 and 1.5 <= death <= 2.5
    assert np.sort(hist[:, 0] - hist[:, 1])[-2] < 1.0


def test_ascent_basins_on_a_plateau_merge_back():
    void = np.zeros((12, 12, 12), bool)
    void[2:10, 2:10, 2:10] = True
    d = V.distance_to_structure(void, margin=8)   # flat top 2 voxels wide
    basins, n = V.ascent_basins(d, void)
    assert (basins > 0).sum() == void.sum()
    merged = V.distance_voids(void, persistence=0.0, margin=8)
    assert merged.max() == 1


def test_min_radius_drops_small_voids():
    n = 40
    void = ball(n, (10, 20, 20), 8) | ball(n, (30, 20, 20), 2)
    labels = V.distance_voids(void, persistence=1.0, min_radius=4.0, margin=16)
    assert labels.max() == 1
    assert labels[30, 20, 20] == 0


def test_threshold_voids_erosion_splits_the_neck():
    void = dumbbell()
    whole = V.threshold_voids(void)
    cut = V.threshold_voids(void, erode=3)
    assert whole.max() == 1
    assert cut.max() == 2
    assert (cut > 0).sum() == void.sum()


def test_threshold_voids_score_threshold():
    void = dumbbell()
    score = np.where(void, 0.9, 0.1)
    score[20:29, :, :] = np.where(void[20:29], 0.55, score[20:29])   # unsure neck
    cut = V.threshold_voids(void, score=score, threshold=0.6)
    assert cut.max() == 2


def test_soft_distance_reduces_to_hard_distance_for_sharp_scores():
    void = dumbbell(32 + 8)
    score = void.astype(np.float32)
    soft = V.soft_distance(score, thresholds=(0.5,), margin=16)
    hard = V.distance_to_structure(void, margin=16)
    assert np.allclose(soft, hard)


def test_sphere_voids_maximal_and_holes():
    n = 64
    cell = 1.0
    void = ball(n, (16, 32, 32), 12) | ball(n, (48, 32, 32), 9)
    void |= ball(n, (16, 32, 44), 6)        # a bump on the first ball
    d = V.distance_to_structure(void, cell=cell, margin=24)
    s = V.sphere_voids(d, void, cell, r_min=8.0, r_hole=3.0)
    assert len(np.unique(s['void'])) == 2
    assert s['maximal'].sum() == 2
    big = s['radius'][s['maximal']].max()
    assert 11 <= big <= 13
    vols = V.sphere_union_volume(s, samples=4000)
    assert vols.max() >= 4 / 3 * np.pi * big ** 3 * 0.95


def test_sphere_overlap_formula():
    # Identical spheres at zero separation: the whole volume.
    assert np.isclose(V._overlap(2.0, 2.0, 0.0), 4 / 3 * np.pi * 8)
    # Touching spheres share nothing.
    assert V._overlap(2.0, 3.0, 5.0) == 0.0
    # Monte Carlo check for a partial overlap.
    rng = np.random.default_rng(1)
    pts = rng.uniform(-3, 5, size=(400000, 3))
    inside = ((pts ** 2).sum(1) < 9) & (((pts - [2.5, 0, 0]) ** 2).sum(1) < 4)
    mc = inside.mean() * 8 ** 3
    assert np.isclose(V._overlap(3.0, 2.0, 2.5), mc, rtol=0.03)


def test_catalog_center_across_a_face():
    n = 32
    labels = ball(n, (0.5, 10, 20), 4).astype(np.int32)   # straddles x = 0
    cat = V.void_catalog(labels, cell=2.0, slab=7)
    cx = cat['center'][0, 0]
    sep = (cx - 1.0 + 32) % 64 - 32
    assert abs(sep) < 0.3
    assert np.allclose(cat['center'][0, 1:], [20.0, 40.0], atol=0.3)
    assert np.isclose(cat['volume'][0], labels.sum() * 8.0)


def test_match_voids_permuted_ids():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 6, size=(12, 12, 12))
    perm = np.array([0, 3, 5, 1, 2, 4])
    m = V.match_voids(a, perm[a])
    assert np.array_equal(m['best'], perm[1:])
    assert np.allclose(m['jaccard'], 1.0)


def test_size_function_and_percolation():
    nf = V.size_function(np.array([1.0, 2.0, 5.0]), 10.0, np.array([0.5, 2.0, 6.0]))
    assert np.allclose(nf, [3, 2, 0] / np.array(1000.0))
    void = np.ones((8, 8, 8), bool)
    void[:, :, 0] = False
    assert V.percolation(void) == {'components': 1, 'largest_share': 1.0}
