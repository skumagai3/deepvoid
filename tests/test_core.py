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
