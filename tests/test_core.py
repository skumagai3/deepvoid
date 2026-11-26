"""Tests for the NumPy parts of the package (no TensorFlow needed)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from deepvoid import Catalog, Volume, load_volume, read_fvol, save_volume, write_fvol
from deepvoid.calibrate import apply_temperature, fit_temperature, reliability
from deepvoid.evaluate import confusion_matrix, scores
from deepvoid.grid import cic
from deepvoid.labels import smoothing_sigma, tweb_eigenvalues, tweb_labels
from deepvoid.observe import select_by_spacing, to_redshift_space
from deepvoid.predict import predict_volume
from deepvoid.preprocess import Normalizer
from deepvoid.sources import Cosmology, Tracers
from deepvoid.split import Region, slab_split

rng = np.random.default_rng(0)


# ---------------------------------------------------------------- io / catalog

def test_fvol_roundtrip(tmp_path):
    a = rng.normal(size=(8, 8, 8)).astype(np.float32)
    write_fvol(a, tmp_path / "a.fvol")
    assert (tmp_path / "a.fvol").stat().st_size == 256 + 4 * a.size
    np.testing.assert_array_equal(read_fvol(tmp_path / "a.fvol"), a)


def test_npz_roundtrip_keeps_metadata_and_dtypes(tmp_path):
    mask = Volume(rng.integers(0, 4, (8, 8, 8)).astype(np.uint8), 10.0, {"field": "mask"})
    save_volume(mask, tmp_path / "m.npz")
    back = load_volume(tmp_path / "m.npz")
    assert back.data.dtype == np.uint8 and back.meta == {"field": "mask"}
    np.testing.assert_array_equal(back.data, mask.data)

    dens = Volume(rng.lognormal(size=(8, 8, 8)).astype(np.float32), 10.0)
    save_volume(dens, tmp_path / "d.npz")
    np.testing.assert_allclose(load_volume(tmp_path / "d.npz").data, dens.data, rtol=1e-3)


def test_float16_overflow_is_refused(tmp_path):
    with pytest.raises(ValueError):
        save_volume(Volume(np.full((4, 4, 4), 1e6, np.float32), 1.0), tmp_path / "x.npz")


def test_catalog_loads_fvol_with_metadata(tmp_path):
    write_fvol(np.ones((4, 4, 4), np.float32), tmp_path / "den.fvol")
    spec = {"root": str(tmp_path),
            "simulations": {"toy": {"sim": "Toy", "box_size": 8.0}},
            "datasets": {"toy/dm": {"file": "den.fvol", "sim": "toy", "field": "density"}}}
    (tmp_path / "cat.yaml").write_text(yaml.safe_dump(spec))
    vol = Catalog(tmp_path / "cat.yaml").load("toy/dm")
    assert vol.box_size == 8.0 and vol.cell_size == 2.0 and vol.meta["sim"] == "Toy"


def test_packaged_catalog_parses():
    cat = Catalog()
    assert "tng300-3/dm" in cat.names() and cat.info("bolshoi/mask")["box_size"] == 250.0


# ---------------------------------------------------------------- gridding / observe

def test_cic_conserves_mass_and_puts_a_point_on_a_grid_point():
    t = Tracers(pos=rng.uniform(0, 10, (500, 3)), box_size=10.0, mass=rng.uniform(1, 2, 500))
    assert np.isclose(cic(t, 16).data.sum(), t.mass.sum(), rtol=1e-5)
    one = Tracers(pos=np.array([[2.5, 5.0, 7.5]]), box_size=10.0, mass=np.array([3.0]))
    grid = cic(one, 4).data  # cell size 2.5: the point sits on grid point (1, 2, 3)
    assert grid[1, 2, 3] == pytest.approx(3.0) and grid.sum() == pytest.approx(3.0)


def test_select_by_spacing_keeps_most_massive():
    t = Tracers(pos=rng.uniform(0, 100, (5000, 3)), box_size=100.0,
                mass=np.arange(5000, dtype=float))
    s = select_by_spacing(t, spacing=10.0)
    assert len(s) == 1000 and s.mass.min() == 4000 and s.spacing == pytest.approx(10.0)


def test_redshift_space_shift_and_wrap():
    t = Tracers(pos=np.array([[1.0, 1.0, 99.0]]), box_size=100.0,
                vel=np.array([[0.0, 0.0, 300.0]]),
                meta={"redshift": 0.0, "cosmology": Cosmology(0.3, 0.7)})
    s = to_redshift_space(t, axis=2)
    assert s.pos[0, 2] == pytest.approx(2.0)  # 99 + 300 km/s / 100 = 102 -> wraps to 2
    assert s.meta["rsd"] and t.pos[0, 2] == 99.0


# ---------------------------------------------------------------- labels

def test_smoothing_sigma_matches_dissertation_convention():
    # TNG300 at 512^3: 0.4 h^-1 Mpc cells -> sigma = sqrt(1 - 0.4^2) h^-1 Mpc
    s = smoothing_sigma(205.0, 512)
    assert s * 205 / 512 == pytest.approx(np.sqrt(1 - (205 / 512) ** 2))
    with pytest.raises(ValueError):
        smoothing_sigma(1000.0, 512)  # Indra at 512^3 is too coarse


def _legacy_eigenvalues(d):
    """Mark's geteigsmark.eigval3dfromden, kept here as the reference."""
    n = d.shape[0]
    dk = np.fft.rfftn(d).astype(np.complex64)
    k1 = np.arange(n).astype(float); k1[k1 > n / 2] -= n
    kz1 = np.arange(n // 2 + 1).astype(float)
    ky, kx, kz = np.meshgrid(k1, k1, kz1)
    k2 = kx ** 2 + ky ** 2 + kz ** 2; k2[0, 0, 0] = 1
    T = np.zeros(d.shape + (3, 3), np.float32)
    for (i, j), (a, b) in {(0, 0): (kx, kx), (1, 0): (kx, ky), (1, 1): (ky, ky),
                           (2, 0): (kx, kz), (2, 1): (ky, kz), (2, 2): (kz, kz)}.items():
        T[..., i, j] = np.fft.irfftn(dk * a * b / k2)
    return np.linalg.eigvalsh(T)


def test_tweb_matches_legacy_code():
    from scipy import ndimage
    den = rng.lognormal(size=(24, 24, 24)).astype(np.float32)
    ref = _legacy_eigenvalues(ndimage.gaussian_filter(den / den.mean(), 1.5, mode="wrap"))
    np.testing.assert_allclose(tweb_eigenvalues(den, 1.5), ref, atol=1e-5)


def test_tweb_labels_are_classes_with_metadata():
    den = Volume(rng.lognormal(size=(16, 16, 16)).astype(np.float32), 8.0, {"sim": "toy"})
    lab = tweb_labels(den)
    assert lab.data.dtype == np.uint8 and lab.data.max() <= 3
    assert lab.meta["sigma_eff"] == pytest.approx(1.0) and lab.meta["sim"] == "toy"


def test_tweb_shift():
    from scipy import ndimage
    den = rng.lognormal(size=(24, 24, 24)).astype(np.float32)
    ref = tweb_eigenvalues(den, 1.5)
    # an integer shift is a roll; a half-cell shift and back is the identity
    np.testing.assert_allclose(tweb_eigenvalues(den, 1.5, shift=(1, 0, 2)),
                               np.roll(ref, (-1, 0, -2), axis=(0, 1, 2)), atol=1e-5)
    half = tweb_eigenvalues(den, 1.5, shift=(0.5, 0.5, 0.5))
    assert np.abs(half - ref).max() > 1e-3
    smooth = ndimage.gaussian_filter(den / den.mean(), 1.5, mode="wrap")
    k = np.fft.fftfreq(24) * 24
    ph = np.exp(2j * np.pi * (k[:, None, None] + k[None, :, None] + k[None, None, :]) * 0.5 / 24)
    moved = np.fft.ifftn(np.fft.fftn(smooth) * ph).real.astype(np.float32)
    np.testing.assert_allclose(half, tweb_eigenvalues(moved, 0.0), atol=2e-3)


# ---------------------------------------------------------------- split / normalization

def test_slab_split_is_disjoint_with_buffers():
    r = slab_split(512, buffer=16)
    assert r["train"].start == 0 and r["val"].start == r["train"].stop + 16
    assert r["test"].start == r["val"].stop + 16 and r["test"].stop == 512 - 16
    masks = [x.mask(64) for x in slab_split(64, buffer=4).values()]
    assert not np.any(masks[0] & masks[1]) and not np.any(masks[1] & masks[2])


def test_crop_origins_stay_inside_region():
    r = slab_split(128, buffer=8)["val"]
    along = r.crop_origins(16, 128)[r.axis]
    assert along.start == r.start and along.stop - 1 + 16 == r.stop


def test_normalizer_fits_on_region_only():
    den = rng.lognormal(size=(16, 16, 16)).astype(np.float32)
    region = slab_split(16, buffer=1)["train"].mask(16)
    norm = Normalizer.fit(den, region)
    x = norm(den)
    assert abs(x[region].mean()) < 1e-4 and abs(x[region].std() - 1) < 1e-4


# ---------------------------------------------------------------- predict / evaluate / calibrate

def test_predict_volume_covers_every_voxel_with_wrap():
    x = rng.integers(0, 4, (40, 40, 40)).astype(np.float32)
    onehot = lambda b: np.eye(4, dtype=np.float32)[b[..., 0].astype(int)]
    probs = predict_volume(onehot, x, crop=24, margin=4, batch_size=3)
    np.testing.assert_array_equal(probs.argmax(-1), x)


def test_scores_on_known_confusion_matrix():
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    perfect = scores(confusion_matrix(y, y))
    assert perfect["mcc"] == pytest.approx(1.0) and perfect["void_f1"] == 1.0
    cm = confusion_matrix(y, np.array([0, 1, 1, 1, 2, 2, 3, 3]))
    s = scores(cm)
    assert s["void_recall"] == 0.5 and s["wall_precision"] == pytest.approx(2 / 3)
    assert 0 < s["mcc"] < 1


def test_temperature_scaling_fixes_overconfidence():
    n = 20000
    labels = rng.integers(0, 4, n)
    logits = rng.normal(size=(n, 4))
    logits[np.arange(n), labels] += 1.0
    true_probs = apply_temperature(np.exp(logits) / np.exp(logits).sum(1, keepdims=True), 1.0)
    overconfident = apply_temperature(true_probs, 0.5)
    t = fit_temperature(overconfident, labels)
    assert t == pytest.approx(2.0, rel=0.1)
    assert reliability(apply_temperature(overconfident, t), labels)["ece"] < \
        reliability(overconfident, labels)["ece"]


# ---------------------------------------------------------------- config / tta / class weights

def test_config_fills_defaults_and_rejects_typos():
    from deepvoid.config import load_config
    cfg = load_config({"name": "x", "data": {"inputs": ["a"], "labels": "b"},
                       "training": {"crop": 64}})
    assert cfg["training"]["crop"] == 64 and cfg["training"]["batch_size"] == 8
    assert cfg["training"]["optimizer"]["name"] == "adamw"
    with pytest.raises(KeyError):
        load_config({"name": "x", "data": {"inputs": ["a"], "labels": "b"},
                     "training": {"patience": 5}})


def test_packaged_baseline_config_is_valid():
    from pathlib import Path
    from deepvoid.config import load_config
    cfg = load_config(Path(__file__).parents[1] / "configs" / "research" / "tng_dm_baseline.yaml")
    assert cfg["training"]["loss"]["class_weights"] == "auto"


def test_tta_is_exact_for_an_equivariant_model():
    from deepvoid.predict import symmetries, with_tta
    assert len(symmetries("flips")) == 8 and len(symmetries("full")) == 48
    onehot = lambda b: np.eye(4, dtype=np.float32)[b[..., 0].astype(int)]
    batch = rng.integers(0, 4, (2, 6, 6, 6, 1)).astype(np.float32)
    np.testing.assert_allclose(with_tta(onehot, "full")(batch), onehot(batch))


def test_tta_undoes_each_transform():
    from deepvoid.predict import with_tta
    # A model that depends on position is not equivariant, so averaging over
    # flips should blur it; undoing transforms correctly keeps the mean intact.
    ramp = lambda b: np.repeat(np.linspace(0, 1, b.shape[1])[None, :, None, None, None],
                               b.shape[0], 0) * np.ones(b.shape[:4] + (4,))
    out = with_tta(ramp, "flips")(np.zeros((1, 8, 8, 8, 1)))
    np.testing.assert_allclose(out, 0.5)


def test_auto_class_weights_balance_to_one():
    from deepvoid.preprocess import auto_class_weights
    labels = np.repeat(np.arange(4), [650, 260, 85, 5])
    w = np.array(auto_class_weights(labels))
    f = np.bincount(labels) / labels.size
    assert np.isclose(np.sum(f * w), 1.0) and w[3] > w[2] > w[1] > w[0]
    assert w[3] / w[0] == pytest.approx(np.sqrt(650 / 5))


def test_default_crops_fit_their_slabs_for_a_512_box():
    from deepvoid.config import DEFAULTS
    tr = DEFAULTS["training"]
    regions = slab_split(512, **DEFAULTS["data"]["split"])
    regions["train"].crop_origins(tr["crop"], 512)
    regions["val"].crop_origins(tr["val_crop"], 512)


# ---------------------------------------------------------------- curriculum / subhalo configs

def test_curriculum_moves_from_start_to_end_weights():
    from deepvoid.curriculum import input_probabilities
    np.testing.assert_allclose(input_probabilities(None, 4, 7), 0.25)
    c = {"start": [4, 3, 2, 1], "end": [1, 1, 1, 1], "epochs": 30}
    np.testing.assert_allclose(input_probabilities(c, 4, 0), [0.4, 0.3, 0.2, 0.1])
    np.testing.assert_allclose(input_probabilities(c, 4, 15), [0.325, 0.275, 0.225, 0.175])
    np.testing.assert_allclose(input_probabilities(c, 4, 30), 0.25)
    np.testing.assert_allclose(input_probabilities(c, 4, 100), 0.25)
    with pytest.raises(ValueError):
        input_probabilities({"start": [1, 0, 1, 1]}, 4, 0)
    with pytest.raises(ValueError):
        input_probabilities({"start": [1, 1]}, 4, 0)


def test_subhalo_configs_are_valid_and_inputs_have_spacings():
    from pathlib import Path
    from deepvoid.config import load_config
    catalog = Catalog()
    paths = sorted((Path(__file__).parents[1] / "configs" / "research" / "subhalo").glob("*.yaml"))
    assert len(paths) == 15
    for path in paths:
        cfg = load_config(path)
        assert catalog.info(cfg["data"]["labels"])["field"] == "mask"
        names = cfg["data"]["inputs"] + (cfg["evaluate"]["inputs"] or [])
        for name in names:
            assert catalog.info(name)["spacing"] > 0
        if cfg["training"]["curriculum"]:
            from deepvoid.curriculum import input_probabilities
            input_probabilities(cfg["training"]["curriculum"], len(cfg["data"]["inputs"]), 0)


def test_stopper_ignores_tiny_gains_but_keeps_the_best_model():
    from deepvoid.stopping import Stopper
    s = Stopper("max", patience=3, min_delta=0.01)
    assert s.update(0, 0.50)                      # first score is the best
    for epoch, v in enumerate([0.501, 0.502, 0.503], start=1):
        assert s.update(epoch, v)                 # still saved as best...
    assert s.should_stop and s.best == 0.503      # ...but too small to reset patience
    s = Stopper("max", patience=3, min_delta=0.01, state=s.state() | {"wait": 0})
    s.update(4, 0.52)
    assert s.wait == 0 and s.ref == 0.52
    loss = Stopper("min", patience=2, min_delta=0.1)
    loss.update(0, 1.0); loss.update(1, 0.95)
    assert loss.best == 0.95 and loss.wait == 1


# ---------------------------------------------------------------- coarse grid

def test_block_mean_and_upsampled_argmax():
    from deepvoid.resample import block_mean, upsampled_argmax
    x = rng.normal(size=(8, 8, 8))
    c = block_mean(x, 2)
    assert c.shape == (4, 4, 4) and np.isclose(c.mean(), x.mean())
    assert np.isclose(c[1, 2, 3], x[2:4, 4:6, 6:8].mean())
    # A coarse cell's class fills all of its fine cells, even a lone one.
    probs = np.full((4, 4, 4, 4), 0.1)
    probs[..., 0] = 0.7
    probs[1, 1, 1] = [0.1, 0.1, 0.7, 0.1]
    fine = upsampled_argmax(probs, 2)
    assert fine.shape == (8, 8, 8)
    assert (fine[2:4, 2:4, 2:4] == 2).all() and fine[0, 0, 0] == 0 and fine[6, 6, 6] == 0


def test_coarse_split_matches_full_resolution_split():
    a = slab_split(512, buffer=16)
    b = slab_split(256, buffer=8)
    for k in a:
        assert (b[k].start * 2, b[k].stop * 2) == (a[k].start, a[k].stop)


def test_coarse_config_and_catalog_entries():
    from pathlib import Path
    from deepvoid.config import load_config
    cfg = load_config(Path(__file__).parents[1] / "configs" / "research" / "subhalo" / "A256_scratch_L10.yaml")
    catalog = Catalog()
    for name in cfg["data"]["inputs"] + cfg["evaluate"]["inputs"] + [cfg["data"]["labels"]]:
        assert "n256" in str(catalog.path(name))
    regions = slab_split(256, **cfg["data"]["split"])
    regions["train"].crop_origins(cfg["training"]["crop"], 256)
    regions["val"].crop_origins(cfg["training"]["val_crop"], 256)


# ---------------------------------------------------------------- input channels

def test_smoothed_channel_fills_a_sparse_field(tmp_path):
    from deepvoid.inputs import build, n_channels, smoothing_cells
    points = np.zeros((32, 32, 32), np.float32)
    points[tuple(rng.integers(0, 32, (3, 20)))] = 1.0       # 20 tracers in 32^3 cells
    write_fvol(points, tmp_path / "pts.fvol")
    spec = {"root": str(tmp_path),
            "simulations": {"toy": {"sim": "Toy", "box_size": 64.0}},
            "datasets": {"toy/pts": {"file": "pts.fvol", "sim": "toy", "field": "density",
                                     "spacing": 8.0}}}
    (tmp_path / "cat.yaml").write_text(yaml.safe_dump(spec))
    catalog = Catalog(tmp_path / "cat.yaml")
    train = slab_split(32, buffer=2)["train"].mask(32)

    assert smoothing_cells({"sigma_spacing": 0.5}, 8.0, 2.0) == 2.0
    norms = {}
    x = build(catalog, "toy/pts", {"sigma_spacing": 0.5}, norms, train)
    assert x.shape == (32, 32, 32, 2) and n_channels({"sigma_spacing": 0.5}) == 2
    assert set(norms) == {"toy/pts", "toy/pts#smooth"}
    raw, smooth = x[..., 0], x[..., 1]
    # The raw channel is one value almost everywhere; the smoothed one varies.
    assert len(np.unique(raw)) <= 3 and len(np.unique(smooth)) > 1000
    # Saved normalization reproduces the same input.
    np.testing.assert_allclose(build(catalog, "toy/pts", {"sigma_spacing": 0.5}, norms), x)
    only = build(catalog, "toy/pts", {"sigma_mpc": 4.0, "keep_raw": False}, {}, train)
    assert only.shape == (32, 32, 32, 1)
    with pytest.raises(KeyError):
        build(catalog, "toy/pts", {"sigma_mpc": 4.0, "typo": 1}, {}, train)


def test_predict_volume_passes_all_channels():
    x = rng.normal(size=(24, 24, 24, 2)).astype(np.float32)
    x[..., 1] = rng.integers(0, 4, (24, 24, 24))
    seen = []
    def fn(b):
        seen.append(b.shape[-1])
        return np.eye(4, dtype=np.float32)[b[..., 1].astype(int)]
    probs = predict_volume(fn, x, crop=16, margin=4, batch_size=2)
    assert set(seen) == {2}
    np.testing.assert_array_equal(probs.argmax(-1), x[..., 1])


# ---------------------------------------------------------------- interpretability helpers

def test_cka_is_one_for_rotated_copies_and_low_for_noise():
    from deepvoid.interpret import linear_cka
    x = rng.normal(size=(500, 8))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    assert linear_cka(x, 3 * x @ q) == pytest.approx(1.0)
    assert linear_cka(x, rng.normal(size=(500, 8))) < 0.1


def test_probe_reads_out_a_linear_target_and_not_noise():
    from deepvoid.interpret import probe_r2
    f = rng.normal(size=(200, 5))
    assert probe_r2(f, f @ np.arange(5.0)) > 0.99
    assert probe_r2(f, rng.normal(size=200)) < 0.2


def test_radial_profile_and_distance_to_tracer():
    from deepvoid.interpret import distance_to_tracer, radial_profile
    cube = np.zeros((9, 9, 9)); cube[4, 4, 4] = 1.0
    r, prof = radial_profile(cube)
    assert r[0] == 0 and prof[0] == 1.0 and prof[1:].sum() == 0
    raw = np.zeros((32, 32, 32)); raw[0, 0, 0] = 1.0           # one tracer at the corner
    region = Region("test", 0, 28, 32)
    d = distance_to_tracer(raw, region, pad=8)
    assert d.shape == (4, 32, 32)
    assert d[3, 0, 0] == pytest.approx(1.0)                     # x=31 wraps to the tracer at x=0
    assert d[0, 0, 0] == pytest.approx(4.0)


def test_binned_scores_counts_and_recall():
    from deepvoid.interpret import binned_scores
    dist = np.array([0.5, 0.5, 1.5, 1.5])
    labels = np.array([0, 1, 0, 1]); pred = np.array([0, 1, 1, 1])
    s = binned_scores(dist, pred, labels, np.array([0, 1, 2]), n_classes=2)
    np.testing.assert_array_equal(s["counts"], [2, 2])
    np.testing.assert_allclose(s["accuracy"], [1.0, 0.5])
    np.testing.assert_allclose(s["recall_0"], [1.0, 0.0])


def test_report_saves_numbers_as_plain_json(tmp_path):
    import json
    from deepvoid.report import Report
    rep = Report(tmp_path / "r")
    rep.record(a=np.float32(0.5), b=np.arange(3), c={3: (np.float64(1.0), np.nan)})
    Report(tmp_path / "r").record(d=1)                       # reopening keeps earlier entries
    saved = json.loads((tmp_path / "r" / "results.json").read_text())
    assert saved == {"a": 0.5, "b": [0, 1, 2], "c": {"3": [1.0, None]}, "d": 1}


def test_evaluation_figure_draws_and_saves(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from deepvoid.evaluate import confusion_matrix, scores
    from deepvoid.figures import evaluation_figure, save_and_display
    truth = rng.integers(0, 4, (32, 32))
    pred = np.where(rng.random((32, 32)) < 0.8, truth, rng.integers(0, 4, (32, 32)))
    cm = confusion_matrix(truth, pred)
    res = {**scores(cm), "confusion_matrix": cm.tolist()}
    rows = [dict(name=f"toy/L{L}", input=rng.normal(size=(32, 32)), truth=truth, pred=pred,
                 result=res) for L in (3, 10)]
    path = save_and_display(evaluation_figure(rows, "toy", box_size=205.0), tmp_path / "f.png")
    assert path.exists() and path.stat().st_size > 10_000


# ---------------------------------------------------------------- void scores and threshold

def test_void_scores_from_confusion_matrix():
    from deepvoid.evaluate import scores
    cm = np.array([[80, 15, 5, 0], [10, 30, 0, 0], [0, 0, 20, 0], [0, 0, 0, 5]])
    s = scores(cm)
    tp, fp, fn = 80, 10, 20
    tn = cm.sum() - tp - fp - fn
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    assert s["void_mcc"] == pytest.approx(mcc)
    assert s["void_iou"] == pytest.approx(tp / (tp + fp + fn))


def test_void_threshold_curve_matches_brute_force_and_decide():
    from deepvoid.evaluate import (THRESHOLDS, best_void_threshold, binary_scores, decide,
                                   void_threshold_curve)
    truth = rng.integers(0, 4, 5000)
    probs = rng.dirichlet(np.ones(4), 5000)
    probs[truth == 0, 0] += 0.3
    probs /= probs.sum(1, keepdims=True)
    curve = void_threshold_curve(probs[:, 0], truth)
    for i in (0, 30, 60, 90):
        t = THRESHOLDS[i]
        called = probs[:, 0] >= t
        ref = binary_scores(np.sum(called & (truth == 0)), np.sum(called & (truth != 0)),
                            np.sum(~called & (truth == 0)), np.sum(~called & (truth != 0)))
        assert curve["mcc"][i] == pytest.approx(float(ref["mcc"]))
    t = best_void_threshold(curve)
    pred = decide(probs, t)
    assert ((pred == 0) == (probs[:, 0] >= t)).all()
    nonvoid = pred != 0
    assert (pred[nonvoid] == probs[nonvoid, 1:].argmax(1) + 1).all()
    assert (decide(probs) == probs.argmax(1)).all()


def test_void_reliability_bins_scores_against_void_fraction():
    from deepvoid.evaluate import void_reliability
    s = rng.random(200_000)
    truth = np.where(rng.random(200_000) < s, 0, 1)          # void with frequency equal to the score
    rel = void_reliability(s, truth)
    assert rel["ece"] < 0.01
    np.testing.assert_allclose(rel["fraction_void"], rel["mean_score"], atol=0.02)
    over = void_reliability(np.clip(s * 1.5, 0, 1), truth)   # scores larger than the void fraction
    assert over["ece"] > 0.05
    assert rel["counts"].sum() == 200_000


def test_precision_recall_readouts_and_pr_figure(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from deepvoid.evaluate import precision_at_recall, recall_at_precision, void_threshold_curve
    from deepvoid.figures import save_and_display, void_pr_figure
    truth = rng.integers(0, 2, 20000)
    score = np.clip(0.5 * (truth == 0) + 0.5 * rng.random(20000), 0, 1)
    c = void_threshold_curve(score, truth)
    p90 = precision_at_recall(c, 0.9)
    assert 0.5 < p90 <= 1.0
    assert recall_at_precision(c, p90) >= 0.9
    assert np.isnan(recall_at_precision(c, 1.01))
    plain_c = {k: np.asarray(v).tolist() for k, v in c.items()}
    fig = void_pr_figure({"runA": {"L10": plain_c, "L3": plain_c}, "runB": {"L10": plain_c}},
                         points={"runA": {"L10": (0.9, p90)}})
    assert save_and_display(fig, tmp_path / "pr.png").exists()


def test_average_precision_limits():
    from deepvoid.evaluate import average_precision, void_threshold_curve
    truth = rng.integers(0, 2, 20000)
    perfect = void_threshold_curve((truth == 0).astype(float) * 0.99 + 0.005, truth)
    assert average_precision(perfect) == pytest.approx(1.0, abs=1e-6)
    blind = void_threshold_curve(rng.random(20000), truth)
    assert average_precision(blind) == pytest.approx((truth == 0).mean(), abs=0.02)


# ---------------------------------------------------------------- runs and figures

def _toy_run(tmp_path):
    from deepvoid.config import load_config, save_config
    out = tmp_path / "toy_run"
    out.mkdir()
    save_config(load_config({"name": "toy_run", "data": {"inputs": ["tng300-3/subhalos/L10"],
                                                          "labels": "tng300-3/mask"}}),
                out / "config.yaml")
    (out / "log.csv").write_text("epoch,loss,val_mcc,val_void_mcc\n0,1.0,0.3,0.5\n1,0.8,0.4,nan\n")
    (out / "test_scores.json").write_text('{"x/L10": {"mcc": 0.6}}')
    (out / "test_scores_margin32.json").write_text('{"x/L10": {"mcc": 0.61}}')
    return out


def test_run_reads_log_and_scores(tmp_path):
    from deepvoid.runs import Run, find_runs
    run = Run(_toy_run(tmp_path))
    assert run.name == "toy_run" and run.finished
    np.testing.assert_array_equal(run.log["epoch"], [0, 1])
    assert np.isnan(run.log["val_void_mcc"][1])
    assert run.scores()["x/L10"]["mcc"] == 0.61
    assert run.scores(prefer="none")["x/L10"]["mcc"] == 0.6
    assert [r.name for r in find_runs(tmp_path)] == ["toy_run"]
    with pytest.raises(FileNotFoundError):
        Run(tmp_path)


def test_figures_draw(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from deepvoid.figures import (fields_figure, prediction_figure, save_and_display,
                                  summary_figure, training_curves)
    from deepvoid.runs import Run
    n = 32
    dm = rng.lognormal(size=(n, n, n))
    truth = rng.integers(0, 4, (n, n, n))
    tracers = (rng.random((n, n, n)) > 0.995).astype(float)
    scores = rng.dirichlet(np.ones(4), (n, n, n))
    figs = [fields_figure({"dm": dm, "coarse": dm[::2, ::2, ::2]}, truth, 5, box_size=64.0),
            training_curves([Run(_toy_run(tmp_path))]),
            prediction_figure(dm, truth, scores, 5, threshold=0.4, box_size=64.0),
            summary_figure(dm, truth, tracers, truth, 5, box_size=64.0, spacing=10, note="toy")]
    for i, fig in enumerate(figs):
        assert save_and_display(fig, tmp_path / f"f{i}.png").exists()


# ---------------------------------------------------------------- point clouds

def test_voxel_index_nearest_grid_point_and_periodic():
    from deepvoid.pointcloud import voxel_index
    pos = np.array([[0.0, 0.49, 0.51], [9.9, 5.0, 0.0]])  # box 10, n 10: cell 1
    np.testing.assert_array_equal(voxel_index(pos, 10, 10.0), [[0, 0, 1], [0, 5, 0]])
    np.testing.assert_array_equal(voxel_index(pos, 10, 10.0, centered=True), [[0, 0, 0], [9, 5, 0]])


def test_compare_with_field_recognizes_its_own_sample():
    from deepvoid.pointcloud import compare_with_field
    t = Tracers(pos=rng.uniform(0, 20, (500, 3)), box_size=20.0, mass=rng.uniform(1, 2, 500))
    same = compare_with_field(t, cic(t, 16).data)
    assert same["correlation_integer"] > 0.999 > same["correlation_centered"]
    other = Tracers(pos=rng.uniform(0, 20, (500, 3)), box_size=20.0, mass=t.mass)
    assert compare_with_field(other, cic(t, 16).data)["correlation_integer"] < 0.5


def test_write_viewer_replaces_viewer_script(tmp_path):
    from deepvoid.pointcloud import write_viewer
    page = tmp_path / "template.html"
    page.write_text('<html><head><title>x</title><script src="libs/potree/potree.js"></script>'
                    '</head><body><script type="module">window.viewer = new Potree.Viewer(a);'
                    '</script></body></html>')
    meta = tmp_path / "pointclouds" / "L3" / "metadata.json"
    out = write_viewer(page, tmp_path / "index.html",
                       [{"name": "L3", "metadata": meta},
                        {"name": "rho", "metadata": meta, "attribute": "log_density"}], title="test")
    html = out.read_text()
    assert "libs/potree/potree.js" in html and "<title>test</title>" in html
    assert '"url": "pointclouds/L3/metadata.json"' in html and "new Potree.Viewer(a)" not in html
    assert '"attribute": "log_density"' in html and '"true wall, predicted void"' in html


def test_void_surface_is_a_shell():
    from deepvoid.pointcloud import void_surface, voxel_positions
    classes = np.ones((10, 10, 10), np.uint8)
    classes[2:7, 2:7, 2:7] = 0                       # a 5^3 void
    surf = void_surface(classes)
    assert surf.sum() == 5 ** 3 - 3 ** 3 and not surf[4, 4, 4] and surf[2, 4, 4]
    np.testing.assert_allclose(voxel_positions(surf, 20.0).min(0), [4.0, 4.0, 4.0])


def test_write_las_roundtrip(tmp_path):
    laspy = pytest.importorskip("laspy")
    from deepvoid.pointcloud import write_las
    pos = rng.uniform(0, 205, (100, 3))
    cls = rng.integers(0, 4, 100)
    write_las(tmp_path / "a.las", pos, cls, extra={"log_mass": rng.normal(size=100).astype(np.float32)})
    las = laspy.read(tmp_path / "a.las")
    np.testing.assert_allclose(np.stack([las.x, las.y, las.z], 1), pos, atol=1e-3)
    np.testing.assert_array_equal(las.classification, cls)
    assert "log_mass" in las.point_format.extra_dimension_names


def test_subcube_split_reproduces_the_paper_protocol_and_a_clean_control():
    from deepvoid.subcubes import extract, copy_fraction, subcube_split
    n, size = 512, 128
    copy = subcube_split(n, size=size, split_by="copy")
    region = subcube_split(n, size=size, split_by="region")
    for s in (copy, region):
        # 48 regions outside the held-out layer, 4 rotations each; 16 test subcubes
        assert len(s["train"]) + len(s["val"]) == 48 * 4
        assert len(s["test"]) == 16
        assert all(e[0] == 384 and e[3] == 0 for e in s["test"])
        assert all(e[0] + size <= 384 for e in s["train"] + s["val"])
        assert not set(s["train"]) & set(s["val"])
    assert len(copy["val"]) == 39            # ceil(0.2 * 192), as sklearn
    assert copy_fraction(region) == 0.0
    assert copy_fraction(copy) > 0.9
    # A rotation is a 90-degree turn about the first two axes, as np.rot90 in the paper
    v = np.arange(n ** 3, dtype=np.int32).reshape(n, n, n)
    cube = extract(v, (128, 0, 256, 1), size)
    assert np.array_equal(cube, np.rot90(v[128:256, 0:128, 256:384], 1, axes=(0, 1)))


def test_papersplit_configs_are_valid():
    from pathlib import Path
    from deepvoid.config import load_config
    from deepvoid.subcubes import settings
    paths = sorted((Path(__file__).parents[1] / "configs" / "research" / "papersplit").glob("*.yaml"))
    assert len(paths) == 4
    for path in paths:
        cfg = load_config(path)
        s = settings(cfg["data"]["subcubes"])
        assert s["split_by"] in path.stem
        Catalog().info(cfg["data"]["inputs"][0])


def test_region_summary_takes_the_spread_over_regions():
    from deepvoid.subcubes import region_summary

    good = np.diag([50, 30, 15, 5])
    worse = good.copy()
    worse[0, 1], worse[0, 0] = 20, 30
    out = region_summary({(0, 0, 0): good, (128, 0, 0): worse, (0, 128, 0): good})["by_region"]
    assert out["regions"] == 3
    assert out["scores"]["mcc"][0] == pytest.approx(1.0)
    assert out["mean"]["mcc"] == pytest.approx(np.mean(out["scores"]["mcc"]))
    expected = np.std(out["scores"]["mcc"], ddof=1) / np.sqrt(3)
    assert out["standard_error"]["mcc"] == pytest.approx(expected)


def test_minmax_scales_to_unit_range_and_round_trips():
    from deepvoid.preprocess import MinMax, normalizer

    den = np.random.default_rng(0).random((8, 8, 8)).astype(np.float32) * 7 + 2
    norm = MinMax.fit(den)
    out = norm(den)
    assert out.min() == pytest.approx(0.0) and out.max() == pytest.approx(1.0)
    again = normalizer(norm.to_dict())
    assert isinstance(again, MinMax)
    np.testing.assert_allclose(again(den), out)


def test_papersplit_configs_use_the_paper_settings():
    from deepvoid.config import load_config

    for name in ("papersplit_L10_copy", "papersplit_L10_region", "papersplit_DM_copy", "papersplit_DM_region"):
        cfg = load_config(f"configs/research/papersplit/{name}.yaml")
        tr = cfg["training"]
        assert cfg["data"]["normalization"] == "minmax"
        assert tr["optimizer"]["name"] == "adam" and tr["optimizer"]["learning_rate"] == 3e-4
        sched = tr["schedule"]
        assert (sched["name"], sched["factor"], sched["patience"]) == ("plateau", 0.25, 10)
        assert tr["ema"] is None and tr["early_stopping"]["monitor"] == "val_loss"


# ---- Project: a user's own simulation ----

def _toy_density(n=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.lognormal(0.0, 1.0, (n, n, n)).astype(np.float32)


def test_project_round_trip_and_config(tmp_path):
    from deepvoid import Catalog, Project

    project = Project(tmp_path / "proj")
    den = _toy_density()
    project.add_density("dm", den, box_size=16.0, tracer="dm")
    labels = project.add_labels("labels", density="dm")          # cells 0.5 < sigma_eff 1
    assert labels.data.dtype == np.uint8 and labels.data.max() <= 3
    assert labels.meta["sigma_eff"] == pytest.approx(1.0)

    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 16.0, (500, 3))
    mass = rng.lognormal(0, 1, 500)
    halos = project.add_tracers("halos/L4", pos, box_size=16.0, n=32, masses=mass, spacing=4.0)
    assert halos.meta["n_tracers"] == 64 and halos.meta["spacing"] == 4.0
    assert halos.data.sum() == pytest.approx(np.sort(mass)[-64:].sum(), rel=1e-5)
    counts = project.add_tracers("halos/all", pos, box_size=16.0, n=32)
    assert counts.data.sum() == pytest.approx(500, rel=1e-5)
    assert counts.meta["spacing"] == pytest.approx(16.0 / 500 ** (1 / 3))

    # A fresh catalog object reads everything back, with paths relative to the file.
    cat = Catalog(project.catalog_file)
    assert cat.names() == ["dm", "halos/L4", "halos/all", "labels"]
    np.testing.assert_allclose(cat.load("dm").data, den)
    assert cat.load("halos/L4").meta["spacing"] == 4.0

    cfg = project.config("first", inputs="halos/L4", labels="labels", model={"depth": 3})
    assert cfg["data"]["inputs"] == ["halos/L4"] and cfg["model"]["depth"] == 3
    assert Catalog.for_config(cfg).names() == cat.names()
    assert Path(cfg["output"]) == (tmp_path / "proj" / "runs").resolve()
    assert "halos/L4" in project.summary()


def test_project_refuses_bad_input(tmp_path):
    from deepvoid import Project

    project = Project(tmp_path / "proj")
    project.add_density("dm", _toy_density(), box_size=16.0)
    with pytest.raises(ValueError):
        project.add_density("neg", -_toy_density(), box_size=16.0)
    with pytest.raises(ValueError):
        project.add_tracers("out", np.full((3, 3), 20.0), box_size=16.0, n=32)
    with pytest.raises(ValueError):
        project.add_mask("bad", np.full((32, 32, 32), 7), box_size=16.0)
    with pytest.raises(ValueError):                      # an input is not a label grid
        project.config("x", inputs="dm", labels="dm")
    with pytest.raises(ValueError):                      # cells of 2 h^-1 Mpc cannot reach sigma_eff = 1
        Project(tmp_path / "coarse").add_density("dm", _toy_density(8), box_size=16.0) \
            and Project(tmp_path / "coarse").add_labels("labels", density="dm")


def test_packaged_catalog_still_loads():
    from deepvoid import Catalog

    cat = Catalog()
    assert "tng300-3/dm" in cat.names() and cat.info("tng300-3/dm")["box_size"] == 205.0


def test_template_config_is_complete_and_valid():
    from deepvoid.config import DEFAULTS, load_config

    path = Path(__file__).parents[1] / "configs" / "template.yaml"
    user = yaml.safe_load(path.read_text())
    cfg = load_config(path)                      # unknown settings would raise
    assert cfg["model"]["depth"] == 5

    def keys(d, prefix=""):
        out = set()
        for k, v in d.items():
            out |= keys(v, f"{prefix}{k}.") if isinstance(v, dict) else {prefix + k}
        return out

    # Every setting is written out, except those that only the first paper's protocol reads.
    written = keys(user)
    missing = {k for k in keys(DEFAULTS)
               if k not in written and not any(w.startswith(k + ".") for w in written)}
    assert missing <= {"training.schedule.factor", "training.schedule.patience",
                       "training.schedule.min_lr"}, missing


def test_find_runs_lists_folders_whose_config_no_longer_loads(tmp_path):
    from deepvoid.runs import find_runs

    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "config.yaml").write_text("name: old\ntraining: {class_weights: auto}\n")
    (tmp_path / "new").mkdir()
    (tmp_path / "new" / "config.yaml").write_text("name: new\ndata: {inputs: [a], labels: b}\n")
    runs = find_runs(tmp_path)
    assert [r.name for r in runs] == ["new", "old"]
    assert not runs[0].finished and not runs[1].finished
    assert runs[0].config["name"] == "new"
    with pytest.raises(KeyError):
        runs[1].config


def test_fetch_downloads_checks_and_writes_a_catalog(tmp_path):
    import hashlib
    from deepvoid import data
    from deepvoid.io import save_volume
    from deepvoid.volume import Volume

    # A stand-in for the Zenodo record: two small files and their manifest.
    record = tmp_path / "record"
    record.mkdir()
    rng = np.random.default_rng(0)
    save_volume(Volume(rng.random((8, 8, 8)).astype(np.float32), 205.0, {"field": "density"}),
                record / data.FILES["tng300-3/subhalos/L10"], dtype="float32")
    save_volume(Volume(rng.integers(0, 4, (8, 8, 8)).astype(np.uint8), 205.0, {"field": "mask"}),
                record / data.FILES["tng300-3/mask"])
    lines = ["# file  bytes  sha256  source"]
    for f in sorted(record.glob("*.npz")):
        lines.append(f"{f.name}  {f.stat().st_size}  {hashlib.sha256(f.read_bytes()).hexdigest()}  x")
    (record / "MANIFEST.txt").write_text("\n".join(lines) + "\n")
    url = record.as_uri() + "/{file}"

    out = tmp_path / "data"
    cat = data.fetch(["tng300-3/subhalos/L10", "tng300-3/mask"], out, url=url)
    assert cat.names() == ["tng300-3/mask", "tng300-3/subhalos/L10"]
    vol = cat.load("tng300-3/subhalos/L10")
    assert vol.box_size == 205.0 and vol.meta["spacing"] == 10.0 and vol.n == 8
    assert cat.load("tng300-3/mask").data.dtype == np.uint8

    # A corrupted local copy is downloaded again; a wrong checksum is refused.
    (out / data.FILES["tng300-3/mask"]).write_bytes(b"broken")
    data.fetch("tng300-3/mask", out, url=url)
    assert cat.load("tng300-3/mask").data.shape == (8, 8, 8)
    (record / data.FILES["tng300-3/mask"]).write_bytes(b"tampered")
    (out / data.FILES["tng300-3/mask"]).unlink()
    with pytest.raises(OSError):
        data.fetch("tng300-3/mask", out, url=url)
    with pytest.raises(KeyError):
        data.fetch("tng300-3/subhalos/L4", out, url=url)


def _fake_pretrained(tmp_path):
    from deepvoid.pretrained import Pretrained
    norms = {s: {"raw": {"mean": -3.0, "std": 0.1 * float(s), "floor": 1e-3},
                 "smooth": {"mean": -0.5, "std": 0.5, "floor": 1e-3}}
             for s in ("3", "5", "7", "10")}
    info = {"cell_size": 0.4, "smoothing": {"sigma_spacing": 0.5, "keep_raw": True},
            "floor": 1e-3, "normalization": norms,
            "evaluate": {"crop": 8, "margin": 2, "batch_size": 2}}
    return Pretrained(tmp_path, info)


def test_pretrained_inputs_match_training_inputs(tmp_path):
    from scipy import ndimage
    from deepvoid.preprocess import normalizer
    model = _fake_pretrained(tmp_path)
    field = np.random.default_rng(1).random((16, 16, 16)).astype(np.float32)
    x = model.inputs(field, box_size=6.4, spacing=6.0)       # nearest trained spacing: 5
    assert x.shape == (16, 16, 16, 2)
    norms = model.info["normalization"]["5"]
    smooth = ndimage.gaussian_filter(field, 0.5 * 6.0 / 0.4, mode="wrap", truncate=4.0)
    np.testing.assert_allclose(x[..., 0], normalizer(norms["raw"])(field), atol=1e-6)
    np.testing.assert_allclose(x[..., 1], normalizer(norms["smooth"])(smooth), atol=1e-6)
    # Refit: each channel standardized on this box.
    xr = model.inputs(field, box_size=6.4, spacing=6.0, stats="refit")
    assert abs(xr[..., 0].mean()) < 1e-4 and abs(xr[..., 0].std() - 1) < 1e-4


def test_pretrained_refuses_what_it_was_not_trained_on(tmp_path):
    from deepvoid import pretrained
    model = _fake_pretrained(tmp_path)
    field = np.ones((16, 16, 16), np.float32)
    with pytest.raises(ValueError, match="cell size"):
        model.inputs(field, box_size=8.0, spacing=5.0)        # 0.5 cells, not 0.4
    model.inputs(field, box_size=16 * 0.39, spacing=5.0)      # 2.5% off: accepted
    with pytest.raises(ValueError, match="trained range"):
        model.inputs(field, box_size=6.4, spacing=12.0)
    model.inputs(field, box_size=6.4, spacing=12.0, stats="refit")
    with pytest.raises(ValueError, match="cube"):
        model.inputs(np.ones((16, 16, 8)), box_size=6.4, spacing=5.0)
    with pytest.raises(KeyError, match="unknown model"):
        pretrained.load("no-such-model")


def test_rotate_setting_and_line_of_sight():
    from deepvoid.config import check_line_of_sight, load_config
    base = {"name": "x", "data": {"inputs": ["a/rsd"], "labels": "m"}}
    with pytest.raises(ValueError, match="rotate"):
        load_config({**base, "training": {"rotate": "xy"}})

    class FakeCatalog:
        def __init__(self, info): self._info = info
        def info(self, name): return self._info

    rsd = FakeCatalog({"rsd": True})
    with pytest.raises(ValueError, match="redshift space"):
        check_line_of_sight(load_config(base), rsd)
    check_line_of_sight(load_config({**base, "training": {"rotate": "z"}}), rsd)
    check_line_of_sight(load_config({**base, "training": {"rotate": False}}), rsd)
    with pytest.raises(ValueError, match="split axis"):         # x is the default split axis
        check_line_of_sight(load_config({**base, "training": {"rotate": "z"}}),
                            FakeCatalog({"rsd": True, "los_axis": 0}))
    check_line_of_sight(load_config({**base, "training": {"rotate": "z"}}),
                        FakeCatalog({"rsd": True, "los_axis": 1}))     # y: swapped onto z
    check_line_of_sight(load_config(base), FakeCatalog({}))     # real space: anything goes


def test_line_of_sight_swap():
    from deepvoid.config import load_config, los_swap
    from deepvoid.observe import los_to_z
    from deepvoid.inputs import align_line_of_sight

    class FakeCatalog:
        def __init__(self, infos): self.infos = infos
        def info(self, name): return self.infos[name]

    cat = FakeCatalog({"real": {}, "z": {"rsd": True, "los_axis": 2},
                       "y": {"rsd": True, "los_axis": 1}, "x": {"rsd": True, "los_axis": 0}})
    cfg = load_config({"name": "t", "data": {"inputs": ["z"], "labels": "m"},
                       "training": {"rotate": "z"}})
    assert los_swap(cfg, cat, "real") is None and los_swap(cfg, cat, "z") is None
    assert los_swap(cfg, cat, "y") == 1
    with pytest.raises(ValueError, match="split axis"):
        los_swap(cfg, cat, "x")
    cfg_z = load_config({"name": "t", "data": {"inputs": ["z"], "labels": "m",
                                               "split": {"axis": 2}}})
    with pytest.raises(ValueError, match="split axis"):
        los_swap(cfg_z, cat, "y")                    # the swap would move the z split

    a = rng.normal(size=(4, 5, 6, 2))
    b = los_to_z(a, 1)
    assert b.shape == (4, 6, 5, 2) and b[1, 3, 2, 1] == a[1, 2, 3, 1]
    np.testing.assert_array_equal(los_to_z(b, 1), a)            # the swap undoes itself
    assert los_to_z(a, 2) is a

    # Inputs seen along y are swapped with their labels; the x slabs stay put.
    n = 8
    labels = rng.integers(0, 4, (n, n, n)).astype(np.uint8)
    xs = [rng.normal(size=(n, n, n, 1)) for _ in range(3)]
    out_x, out_y = align_line_of_sight(cfg, cat, ["real", "z", "y"], xs, labels)
    assert out_x[0] is xs[0] and out_y[0] is labels and out_y[1] is labels
    np.testing.assert_array_equal(out_x[2], np.swapaxes(xs[2], 1, 2))
    np.testing.assert_array_equal(out_y[2], np.swapaxes(labels, 1, 2))
    test = slab_split(n, buffer=0, axis=0)["test"].mask(n)
    np.testing.assert_array_equal(los_to_z(test, 1), test)      # an x slab is unchanged


def test_redshift_space_keeps_columns_along_line_of_sight():
    from deepvoid.grid import cic
    from deepvoid.observe import to_redshift_space
    from deepvoid.sources import Cosmology, Tracers
    rng = np.random.default_rng(3)
    t = Tracers(pos=rng.random((5000, 3)) * 50.0, box_size=50.0, mass=rng.random(5000) + 0.5,
                vel=rng.normal(0, 400, (5000, 3)),
                meta={"cosmology": Cosmology(omega_m=0.3, h=0.7), "redshift": 0.0})
    real, rsd = cic(t, 32).data, cic(to_redshift_space(t, axis=2), 32).data
    np.testing.assert_allclose(real.sum(axis=2), rsd.sum(axis=2), rtol=1e-5, atol=1e-6)
    assert not np.allclose(real.sum(axis=0), rsd.sum(axis=0))
    # At z = 0 the shift is v / (100 km/s) in h^-1 Mpc.
    moved = to_redshift_space(t, axis=2)
    np.testing.assert_allclose((moved.pos[:, 2] - t.pos[:, 2] - t.vel[:, 2] / 100.0 + 25.0) % 50.0,
                               25.0, atol=1e-9)


def test_pretrained_load_downloads_and_checks(tmp_path):
    import hashlib
    import json as _json
    from deepvoid import pretrained
    record = tmp_path / "record"
    record.mkdir()
    (record / "model.json").write_text(_json.dumps({"cell_size": 0.4, "normalization": {}}))
    (record / "model.keras").write_bytes(b"weights")
    lines = ["# file  bytes  sha256"] + [
        f"{f.name}  {f.stat().st_size}  {hashlib.sha256(f.read_bytes()).hexdigest()}"
        for f in sorted(record.iterdir())]
    (record / "MANIFEST.txt").write_text("\n".join(lines) + "\n")
    url = record.as_uri() + "/{file}"
    model = pretrained.load(folder=tmp_path / "models", url=url)
    assert model.info["cell_size"] == 0.4
    assert (tmp_path / "models" / pretrained.DEFAULT / "model.keras").read_bytes() == b"weights"
    (record / "model.keras").write_bytes(b"tampered")              # differs from MANIFEST
    (tmp_path / "models" / pretrained.DEFAULT / "model.keras").unlink()
    with pytest.raises(OSError, match="checksum"):
        pretrained.load(folder=tmp_path / "models", url=url)


def test_input_labels_keep_variants_apart():
    from deepvoid.config import check_unique_labels, input_label
    assert input_label("tng300-3/subhalos/L10") == "L10"
    assert input_label("tng300-3/subhalos/L10/rsd-z") == "L10_rsd-z"
    rsd = [f"tng300-3/subhalos/L{s}/rsd-z" for s in (3, 5, 7, 10)]
    check_unique_labels(rsd)
    with pytest.raises(ValueError, match="share the labels"):
        check_unique_labels(["a/L10", "b/L10"])


# ---------------------------------------------------------------- strip pooling

def test_strip_pool_config_is_validated():
    from deepvoid.config import load_config
    base = {"name": "x", "data": {"inputs": ["a"], "labels": "b"}}
    assert load_config(base)["model"]["strip_pool"] is None
    ok = load_config({**base, "model": {"strip_pool": {"axes": ["z"], "length": 5}}})
    assert ok["model"]["strip_pool"] == {"axes": ["z"], "length": 5}
    for bad in ({"axes": ["z"], "length": 4}, {"axes": ["w"], "length": 5},
                {"axes": ["z", "z"], "length": 5}, {"axes": [], "length": 5},
                {"axes": ["z"]}, {"axes": ["z"], "length": True}):
        with pytest.raises(ValueError):
            load_config({**base, "model": {"strip_pool": bad}})


def test_axis_radii_and_cylinder_map_see_a_stretch_along_z():
    from deepvoid.interpret import axis_radii, cylinder_map
    idx = np.indices((33, 33, 33)) - 16
    blob = np.exp(-0.5 * ((idx[0] / 2) ** 2 + (idx[1] / 2) ** 2 + (idx[2] / 6) ** 2))
    r = axis_radii(blob)
    assert r["x"] == r["y"] and r["z"][0] > 2 * r["x"][0] and r["z"][1] > 2 * r["x"][1]
    iso = np.exp(-0.5 * (idx ** 2).sum(0) / 9)
    ri = axis_radii(iso)
    assert ri["x"] == ri["y"] == ri["z"]
    m = cylinder_map(blob, 10)
    assert m.shape == (10, 10)
    assert m[0, 0] == pytest.approx(1.0)
    assert m[6, 0] > m[0, 6]                    # falls off slower along z than across


def test_strip_pool_starts_as_identity_and_reaches_further_along_z():
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    from deepvoid.models import copy_weights, unet3d
    tf.keras.utils.set_random_seed(0)
    plain = unet3d(depth=3, base_filters=4)
    strip = unet3d(depth=3, base_filters=4, strip_pool={"axes": ["z"], "length": 7})
    copied, skipped = copy_weights(plain, strip)
    assert all(n.startswith("bottom_strip_") for n in skipped)
    x = tf.random.normal((1, 32, 32, 64, 1))
    np.testing.assert_allclose(plain(x), strip(x), atol=1e-5)    # zero-initialized output

    out = strip.get_layer("bottom_strip_out")
    out.set_weights([w + 0.1 * rng.normal(size=w.shape) for w in out.get_weights()])

    def extent(model, axis):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = tf.reduce_sum(model(x)[0, 16, 16, 32] * tf.constant([1.0, -1, 2, -2]))
        g = np.abs(tape.gradient(y, x)[0, ..., 0].numpy())
        line = g.sum(axis=tuple(a for a in range(3) if a != axis))
        nz = np.nonzero(line > 1e-12 * line.max())[0]
        c = x.shape[1 + axis] // 2
        return max(c - nz.min(), nz.max() - c)

    assert extent(strip, 2) > extent(plain, 2)    # reaches further along z
    assert extent(strip, 0) == extent(plain, 0)   # but not across it


# ---------------------------------------------------------------- information limit

def test_majority_by_distance_and_ceiling():
    from deepvoid.interpret import (majority_by_distance, mcc_if_fixed_beyond,
                                    predict_by_distance, region_slice)
    dist = np.array([0.5, 0.6, 1.5, 1.6, 1.7, 9.0, 12.0])
    labels = np.array([3, 3, 2, 2, 1, 0, 0])
    edges = np.array([0.0, 1.0, 2.0, 4.0, 8.0])
    table = majority_by_distance(dist, labels, edges)
    assert table.tolist() == [3, 2, 2, 0]           # empty 2-4 bin takes the bin below
    assert predict_by_distance(np.array([0.1, 3.0, 50.0]), table, edges).tolist() == [3, 2, 0]

    pred = np.array([3, 3, 2, 2, 2, 1, 1])          # wrong at 1.7, 9 and 12
    exact = mcc_if_fixed_beyond(pred, labels, dist, [100.0, 8.0, 1.0])
    from deepvoid.evaluate import confusion_matrix, scores
    assert exact[0] == pytest.approx(scores(confusion_matrix(labels, pred))["mcc"])
    assert exact[0] < exact[1] < exact[2] == pytest.approx(1.0)

    r = Region("test", 1, 2, 5)
    a = np.arange(6 ** 3).reshape(6, 6, 6)
    assert a[region_slice(r)].shape == (6, 3, 6)


def test_gather_crops_match_numpy_slices():
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    from deepvoid.pipeline import _gather_crops
    vols = rng.normal(size=(3, 10, 10, 10, 2)).astype(np.float32)
    which = tf.constant([2, 0], tf.int32)
    origins = tf.constant([[1, 2, 3], [4, 0, 5]], tf.int32)
    out = _gather_crops(tf.constant(vols), which, origins, tf.range(4, dtype=tf.int32)).numpy()
    np.testing.assert_array_equal(out[0], vols[2, 1:5, 2:6, 3:7])
    np.testing.assert_array_equal(out[1], vols[0, 4:8, 0:4, 5:9])


# ---------------------------------------------------------------- SMDPL helpers

def test_onto_coarse_points():
    from deepvoid.resample import onto_coarse_points
    n = 16
    np.testing.assert_allclose(onto_coarse_points(np.full((n, n, n), 3.0), 0), 3.0, rtol=1e-6)
    # A periodic wave is scaled by (1 + cos(2 pi k / n)) / 2 per axis and read exactly at
    # the coarse points: fine 2j + offset.
    k = 2
    x = np.arange(n)
    wave = np.cos(2 * np.pi * k * x / n)
    f = wave[:, None, None] * np.ones((1, n, n))
    for offset in (0, 1):
        got = onto_coarse_points(f, offset)
        want = (1 + np.cos(2 * np.pi * k / n)) / 2 * wave[offset::2]
        np.testing.assert_allclose(got[:, 3, 5], want, atol=1e-6)
    # A spike at fine point 2j + offset lands at coarse point j with weight 1/8.
    for offset in (0, 1):
        spike = np.zeros((n, n, n))
        spike[2 * 3 + offset, 2 * 5 + offset, 2 * 1 + offset] = 1.0
        out = onto_coarse_points(spike, offset)
        assert out[3, 5, 1] == pytest.approx(0.125) and out.argmax() == np.ravel_multi_index((3, 5, 1), out.shape)
    two = np.stack([f, 2 * f], axis=-1)
    np.testing.assert_allclose(onto_coarse_points(two, 1)[..., 1], 2 * onto_coarse_points(f, 1), atol=1e-6)


def test_predict_volume_float16_output():
    from deepvoid.predict import predict_volume
    x = rng.normal(size=(16, 16, 16, 1)).astype(np.float32)
    fn = lambda b: np.repeat(np.full(b.shape[:4] + (1,), 0.25, np.float32), 4, axis=-1)
    out = predict_volume(fn, x, crop=8, margin=2, batch_size=3, out_dtype=np.float16)
    assert out.dtype == np.float16 and out.shape == (16, 16, 16, 4)
    np.testing.assert_allclose(out.astype(np.float32), 0.25)


def test_describe_run_refuses_shared_spacing(tmp_path):
    import json as _json
    from types import SimpleNamespace
    from deepvoid.config import load_config
    from deepvoid.pretrained import describe_run
    names = ["s/L3/rsd-z", "s/L3/rsd-y"]
    cfg = load_config({"name": "r", "data": {"inputs": names, "labels": "m"},
                       "training": {"rotate": "z"}})
    (tmp_path / "normalization.json").write_text(_json.dumps(
        {"s/L3/rsd-z": {"mean": 1.0, "std": 2.0, "floor": 1e-3},
         "s/L3/rsd-y": {"mean": 1.5, "std": 2.5, "floor": 1e-3}}))
    vol = SimpleNamespace(meta={"spacing": 3.0}, cell_size=0.4)
    run = SimpleNamespace(config=cfg, path=tmp_path, name="r",
                          catalog=SimpleNamespace(load=lambda name: vol))
    with pytest.raises(ValueError, match="share spacing"):
        describe_run(run)
    info = describe_run(run, inputs=["s/L3/rsd-z"])
    assert info["normalization"]["3"]["raw"]["mean"] == 1.0 and info["inputs"] == ["s/L3/rsd-z"]
    with pytest.raises(ValueError, match="not inputs"):
        describe_run(run, inputs=["s/L5/rsd-z"])


def test_read_smdpl_halos(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from deepvoid.sources.smdpl import read_halos
    df = pd.DataFrame({"X": [1.0, 401.0], "Y": [2.0, -1.0], "Z": [3.0, 5.0],
                       "VX": [10.0, 0.0], "VY": [0.0, 0.0], "VZ": [-5.0, 7.0],
                       "MVIR": [1e12, 2e12], "UPID": [-1, 7]})
    df.to_parquet(tmp_path / "h.parquet")
    h = read_halos(tmp_path / "h.parquet")
    np.testing.assert_allclose(h.pos[1], [1.0, 399.0, 5.0])
    assert h.box_size == 400.0 and h.meta["n_subhalos"] == 1 and h.mass[1] == 2e12


def test_read_bolshoi_halos(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from deepvoid.sources.bolshoi import read_halos
    df = pd.DataFrame({"x": [1.0, 251.0, 7.0], "y": [2.0, -1.0, 8.0], "z": [3.0, 5.0, 9.0],
                       "vx": [10.0, 0.0, 1.0], "vy": [0.0, 0.0, 1.0], "vz": [-5.0, 7.0, 1.0],
                       "Mvir": [1e12, 2e12, 3e11], "hostFlag": [-1, 416000000001, -1]})
    df.to_parquet(tmp_path / "h.parquet")
    h = read_halos(tmp_path / "h.parquet")
    np.testing.assert_allclose(h.pos[1], [1.0, 249.0, 5.0])
    assert h.box_size == 250.0 and h.meta["n_subhalos"] == 1 and len(h.mass) == 3
    d = read_halos(tmp_path / "h.parquet", distinct_only=True)
    assert len(d.mass) == 2 and d.meta["n_subhalos"] == 0 and d.mass[1] == 3e11
    with pytest.raises(ValueError, match="hostflag"):
        df.drop(columns="hostFlag").to_parquet(tmp_path / "g.parquet")
        read_halos(tmp_path / "g.parquet", distinct_only=True)


def test_catalog_crop(tmp_path):
    from deepvoid.io import save_volume, write_fvol
    rng = np.random.default_rng(0)
    real = rng.random((12, 12, 12)).astype(np.float32)
    padded = np.pad(real, ((0, 4),) * 3, mode="wrap")
    write_fvol(padded, tmp_path / "d.fvol")
    save_volume(Volume(data=padded, box_size=50.0, meta={}), tmp_path / "d.npz", dtype="float32")
    (tmp_path / "cat.yaml").write_text(
        "simulations:\n  s:\n    box_size: 50.0\n"
        "datasets:\n"
        "  a: {file: d.fvol, sim: s, field: density, crop: 12}\n"
        "  b: {file: d.npz, sim: s, field: density, crop: 12}\n"
        "  c: {file: d.fvol, sim: s, field: density}\n"
        "  e: {file: d.fvol, sim: s, field: density, crop: 20}\n")
    cat = Catalog(tmp_path / "cat.yaml")
    for name in ("a", "b"):
        v = cat.load(name)
        assert v.data.shape == (12, 12, 12) and v.box_size == 50.0
        np.testing.assert_array_equal(v.data, real)
        assert v.meta["cropped_from"] == 16
    assert cat.load("c").data.shape == (16, 16, 16)
    with pytest.raises(ValueError, match="crop"):
        cat.load("e")


def test_packaged_bolshoi_is_624():
    cat = Catalog()
    for name in ("bolshoi/dm", "bolshoi/halos/L3", "bolshoi/halos/L10", "bolshoi/mask-legacy"):
        assert cat.info(name)["crop"] == 624
    assert "crop" not in cat.info("bolshoi/n640-padded/dm")
    assert cat.info("bolshoi/mask")["box_size"] == 250.0
