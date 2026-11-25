"""Tests for the V-web labels, the velocity grid and the TNG snapshot reader."""

import numpy as np
import pytest
import scipy.fft as sfft

from deepvoid.grid import velocity_field
from deepvoid.labels import tweb_eigenvalues, vweb_eigenvalues, vweb_labels
from deepvoid.labels.tensor import symmetric_eigenvalues
from deepvoid.labels.vweb import H0
from deepvoid.sources import Tracers
from deepvoid.volume import Volume

rng = np.random.default_rng(0)


def test_symmetric_eigenvalues_match_full_eigvalsh():
    n = 6
    a = rng.normal(size=(n, n, n, 3, 3)).astype(np.float32)
    full = 0.5 * (a + np.swapaxes(a, -1, -2))
    comps = {(i, j): full[..., i, j] for i in range(3) for j in range(i + 1)}
    np.testing.assert_allclose(symmetric_eigenvalues(comps, slab=4),
                               np.linalg.eigvalsh(full), atol=1e-5)


def test_vweb_of_a_pure_shear_flow():
    # v_x = A sin(2 pi y / L): Sigma_xy = Sigma_yx = -(A k / 2 H0) cos(k y), all else 0,
    # so the eigenvalues are -|Sigma_xy|, 0, +|Sigma_xy|.
    n, box, amp = 32, 64.0, 300.0
    k = 2 * np.pi / box
    y = np.arange(n) * box / n
    vx = np.broadcast_to(amp * np.sin(k * y)[None, :, None], (n, n, n))
    zero = np.zeros((n, n, n))
    eig = vweb_eigenvalues([vx, zero, zero], box)
    s = np.abs(amp * k / (2 * H0) * np.cos(k * y))[None, :, None]
    np.testing.assert_allclose(eig[..., 0], -np.broadcast_to(s, (n, n, n)), atol=1e-4)
    np.testing.assert_allclose(eig[..., 1], 0.0, atol=1e-4)
    np.testing.assert_allclose(eig[..., 2], np.broadcast_to(s, (n, n, n)), atol=1e-4)


def test_vweb_sign_convergent_flow_is_positive():
    # v_x = -A sin(k x) converges onto x = 0: Sigma_xx = A k cos(k x) / H0 > 0 there.
    n, box, amp = 32, 64.0, 200.0
    k = 2 * np.pi / box
    x = np.arange(n) * box / n
    vx = np.broadcast_to((-amp * np.sin(k * x))[:, None, None], (n, n, n))
    zero = np.zeros((n, n, n))
    eig = vweb_eigenvalues([vx, zero, zero], box)
    np.testing.assert_allclose(eig[0, 0, 0, 2], amp * k / H0, rtol=1e-4)


def test_vweb_equals_f_times_tweb_in_linear_theory():
    # Linear theory: v = -f H0 grad(phi) with del^2 phi = delta, so Sigma = f T exactly.
    n, box, f = 32, 100.0, 0.52
    delta = rng.normal(size=(n, n, n))
    d_k = sfft.fftn(delta)
    d_k[n // 2, :, :] = d_k[:, n // 2, :] = d_k[:, :, n // 2] = 0  # no Nyquist modes
    d_k[0, 0, 0] = 0
    delta = sfft.ifftn(d_k).real
    k1 = 2 * np.pi / box * sfft.fftfreq(n) * n
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k2[0, 0, 0] = 1.0
    phi_k = -sfft.fftn(delta) / k2
    vel = [sfft.ifftn(-f * H0 * 1j * kk * phi_k).real for kk in (kx, ky, kz)]
    np.testing.assert_allclose(vweb_eigenvalues(vel, box),
                               f * tweb_eigenvalues(1.0 + delta, 0.0), atol=2e-5)


def test_vweb_labels_need_a_threshold_and_count_eigenvalues():
    n, box = 16, 32.0
    vel = [Volume(rng.normal(scale=50.0, size=(n, n, n)).astype(np.float32), box,
                  {"field": "velocity", "component": c, "sigma_eff": 1.0}) for c in "xyz"]
    with pytest.raises(TypeError):
        vweb_labels(vel)
    lab = vweb_labels(vel, 0.2)
    eig = vweb_eigenvalues([v.data for v in vel], box)
    np.testing.assert_array_equal(lab.data, (eig > 0.2).sum(-1))
    assert lab.meta["labels"] == "vweb" and lab.meta["threshold"] == 0.2
    assert lab.meta["sigma_eff"] == 1.0 and "component" not in lab.meta


def _particles(m, box, vel):
    return Tracers(pos=rng.uniform(0, box, (m, 3)), box_size=box, vel=vel)


def test_velocity_field_recovers_a_uniform_flow():
    box, n, m = 32.0, 16, 20_000
    v0 = np.array([300.0, -120.0, 45.0])
    t = _particles(m, box, np.tile(v0, (m, 1)))
    vel, count = velocity_field(t, n, sigma_eff=3.0, return_count=True)
    for i in range(3):
        np.testing.assert_allclose(vel[i].data, v0[i], rtol=1e-4)
    assert vel[0].meta["empty_cells"] == 0
    assert vel[0].meta["sigma_eff"] == pytest.approx(3.0)
    assert count.data.sum() == pytest.approx(m, rel=1e-4)


def test_velocity_field_from_chunks_equals_all_at_once():
    box, n, m = 32.0, 16, 6_000
    t = _particles(m, box, rng.normal(scale=100.0, size=(m, 3)))
    parts = [Tracers(pos=t.pos[s], box_size=box, vel=t.vel[s])
             for s in (slice(0, 2_000), slice(2_000, m))]
    whole = velocity_field(t, n, sigma_cells=1.5)
    split = velocity_field(iter(parts), n, sigma_cells=1.5)
    for a, b in zip(whole, split):
        np.testing.assert_allclose(a.data, b.data, rtol=1e-4, atol=1e-3)


def test_velocity_field_marks_cells_with_no_particles():
    box, n = 32.0, 16
    t = Tracers(pos=np.full((10, 3), 1.0), box_size=box, vel=np.ones((10, 3)))
    vel = velocity_field(t, n, sigma_cells=0.0)
    assert vel[0].meta["empty_cells"] == n ** 3 - 8
    assert np.count_nonzero(vel[0].data) == 8


def test_read_snapshot_dm_converts_units(tmp_path):
    h5py = pytest.importorskip("h5py")
    from deepvoid.sources.tng import read_snapshot_dm, snapshot_files

    a = 0.25
    pos_kpc, vel_raw, ids = [], [], []
    for chunk, m in enumerate((5, 7)):
        p = rng.uniform(0, 205_000.0, (m, 3))
        v = rng.normal(scale=300.0, size=(m, 3))
        i = np.arange(m, dtype=np.uint64) + 100 * chunk
        with h5py.File(tmp_path / f"snap_099.{chunk}.hdf5", "w") as h:
            head = h.create_group("Header")
            head.attrs["BoxSize"] = 205_000.0
            head.attrs["Time"] = a
            head.attrs["MassTable"] = np.array([0, 0.5, 0, 0, 0, 0])
            head.attrs["NumPart_ThisFile"] = np.array([0, m, 0, 0, 0, 0])
            dm = h.create_group("PartType1")
            dm["Coordinates"], dm["Velocities"], dm["ParticleIDs"] = p, v, i
        pos_kpc.append(p), vel_raw.append(v), ids.append(i)
    assert [f.name for f in snapshot_files(tmp_path)] == ["snap_099.0.hdf5", "snap_099.1.hdf5"]
    t = read_snapshot_dm(tmp_path, ids=True)
    assert t.box_size == 205.0 and len(t) == 12
    np.testing.assert_allclose(t.pos, np.concatenate(pos_kpc) / 1000.0, rtol=1e-6)
    np.testing.assert_allclose(t.vel, np.concatenate(vel_raw) * np.sqrt(a), rtol=1e-6)
    np.testing.assert_array_equal(t.meta["ids"], np.concatenate(ids))
    assert t.meta["redshift"] == pytest.approx(3.0)
    assert t.meta["particle_mass"] == pytest.approx(0.5e10)
