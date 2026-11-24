"""Tests for distillation: the loss, the sampler's teacher crops and the config."""

import numpy as np
import pytest

from deepvoid.config import load_config

BASE = {'name': 't', 'data': {'inputs': ['a'], 'labels': 'm'}}


def with_distill(d, **training):
    return {**BASE, 'training': {'distill': d, **training}}


def test_config_defaults_and_checks():
    cfg = load_config(with_distill({'teacher': 'dm_run'}))
    assert cfg['training']['distill'] == {'teacher': 'dm_run', 'alpha': 0.5, 'temperature': 2.0}
    assert load_config(BASE)['training']['distill'] is None
    for bad in ({'alpha': 0.5}, {'teacher': 'x', 'alpha': 1.5},
                {'teacher': 'x', 'temperature': 0}, {'teacher': 'x', 'beta': 1}):
        with pytest.raises(ValueError):
            load_config(with_distill(bad))
    with pytest.raises(ValueError):
        load_config({**with_distill({'teacher': 'x'}), 'model': {'deep_supervision': True}})


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def packed(label, q):
    return np.concatenate([label[..., None].astype(np.float32), q.astype(np.float32)], -1)


def test_distill_loss_reference_values():
    pytest.importorskip('tensorflow')
    from deepvoid.losses import DistillLoss, WeightedSCCE
    rng = np.random.default_rng(0)
    p = softmax(rng.normal(size=(2, 3, 3, 3, 4))).astype(np.float32)
    q = softmax(rng.normal(size=(2, 3, 3, 3, 4))).astype(np.float32)
    label = rng.integers(0, 4, size=(2, 3, 3, 3))
    w = [0.5, 1.0, 2.0, 4.0]
    y = packed(label, q)

    # alpha = 1: the plain weighted cross-entropy.
    a = np.asarray(DistillLoss(w, alpha=1.0, reduction=None)(y, p))
    b = np.asarray(WeightedSCCE(w, reduction=None)(label, p))
    assert np.allclose(a, b, atol=1e-5)

    # alpha = 0: T^2 KL between the softened distributions, by hand.
    t = 3.0
    pt, qt = softmax(np.log(p) / t), softmax(np.log(q) / t)
    kl = (qt * (np.log(qt) - np.log(pt))).sum(-1)
    want = t * t * kl * np.take(w, label)
    got = np.asarray(DistillLoss(w, alpha=0.0, temperature=t, reduction=None)(y, p))
    assert np.allclose(got, want, atol=1e-5)

    # A student that matches the teacher pays nothing for the soft term.
    same = np.asarray(DistillLoss(None, alpha=0.0, reduction=None)(packed(label, p), p))
    assert np.allclose(same, 0.0, atol=1e-5)


def test_distill_loss_gradient_and_serialization():
    tf = pytest.importorskip('tensorflow')
    import keras
    from deepvoid.losses import DistillLoss
    rng = np.random.default_rng(1)
    logits = tf.Variable(rng.normal(size=(1, 4, 4, 4, 4)).astype(np.float32))
    q = softmax(rng.normal(size=(1, 4, 4, 4, 4)))
    y = packed(rng.integers(0, 4, size=(1, 4, 4, 4)), q)
    loss = DistillLoss([1, 1, 2, 3], alpha=0.0, temperature=2.0)
    with tf.GradientTape() as tape:
        value = loss(y, tf.nn.softmax(logits))
    grad = tape.gradient(value, logits)
    assert float(tf.reduce_max(tf.abs(grad))) > 0
    again = keras.losses.deserialize(keras.losses.serialize(loss))
    assert again.get_config() == loss.get_config()


def test_sampler_teacher_crops_follow_the_labels():
    pytest.importorskip('tensorflow')
    from deepvoid.pipeline import CropSampler
    from deepvoid.split import slab_split
    rng = np.random.default_rng(2)
    n = 32
    labels = rng.integers(0, 4, size=(n, n, n)).astype(np.uint8)
    # Teacher scores that are a known function of the label: argmax = label.
    soft = np.full((n, n, n, 4), 0.1, np.float32)
    np.put_along_axis(soft, labels[..., None].astype(int), 0.7, axis=-1)
    x = rng.normal(size=(n, n, n)).astype(np.float32)
    region = slab_split(n, fractions=(0.7, 0.15, 0.15), buffer=2)['train']
    plain = CropSampler([x, x], [labels, labels], region, crop=8, batch_size=4,
                        steps_per_epoch=3, rotate=True, seed=5)
    dist = CropSampler([x, x], [labels, labels], region, crop=8, batch_size=4,
                       steps_per_epoch=3, rotate=True, seed=5, soft=soft)
    for i in range(3):
        xa, ya = plain[i]
        xb, yb = dist[i]
        assert np.array_equal(np.asarray(xa), np.asarray(xb))      # same crops as without
        yb = np.asarray(yb)
        assert yb.shape == (4, 8, 8, 8, 5) and yb.dtype == np.float32
        assert np.array_equal(yb[..., 0].astype(int), np.asarray(ya))
        assert np.array_equal(yb[..., 1:].argmax(-1), np.asarray(ya))   # turned together
    with pytest.raises(ValueError):
        CropSampler([x], [labels], region, crop=8, soft=soft[:16])
