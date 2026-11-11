"""Tests for the NumPy parts of the package (no TensorFlow needed)."""

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
from deepvoid.split import slab_split

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
    cfg = load_config(Path(__file__).parents[1] / "configs" / "tng_dm_baseline.yaml")
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
    paths = sorted((Path(__file__).parents[1] / "configs" / "subhalo").glob("*.yaml"))
    assert len(paths) == 7
    for path in paths:
        cfg = load_config(path)
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
    cfg = load_config(Path(__file__).parents[1] / "configs" / "subhalo" / "A256_scratch_L10.yaml")
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
