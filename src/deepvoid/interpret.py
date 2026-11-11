"""Tools for looking inside a trained model.

Used by ``notebooks/08_interpretability.ipynb``. The NumPy functions work
without TensorFlow; the ones that run a model import it when called.

- :func:`linear_cka`: how similar two sets of activations are (Kornblith et
  al. 2019), invariant to rotations and scaling of the feature space.
- :func:`probe_r2`: how well a quantity (e.g. log tracer spacing) can be
  read out of features by ridge regression, cross-validated.
- :func:`input_gradient` and :func:`radial_profile`: the effective
  receptive field, i.e. how strongly the prediction at one voxel depends on
  the input at each distance from it.
- :func:`distance_to_tracer` and :func:`binned_scores`: accuracy as a
  function of distance to the nearest tracer.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .split import Region


# ---------------------------------------------------------------- NumPy only

def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between activations x (n, p) and y (n, q) of the same n samples.

    1 means the two representations are the same up to rotation and
    scaling; values near 0 mean unrelated.
    """
    x = x - x.mean(0)
    y = y - y.mean(0)
    xty = np.linalg.norm(y.T @ x) ** 2
    return float(xty / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y) + 1e-12))


def probe_r2(features: np.ndarray, target: np.ndarray, folds: int = 5,
             alphas=(1e-2, 1e-1, 1.0, 10.0, 100.0), seed: int = 0) -> float:
    """Cross-validated R^2 of a ridge regression from features (n, p) to target (n,).

    Features are standardized; the best ridge strength is picked by the
    same cross-validation, so the result is mildly optimistic for very
    small n.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(target))
    f = features[order].astype(np.float64)
    t = target[order].astype(np.float64)
    parts = np.array_split(np.arange(len(t)), folds)
    best = -np.inf
    for alpha in alphas:
        pred = np.empty_like(t)
        for test in parts:
            train = np.setdiff1d(np.arange(len(t)), test)
            mu, sd = f[train].mean(0), f[train].std(0) + 1e-8
            a, b = (f[train] - mu) / sd, (f[test] - mu) / sd
            tm = t[train].mean()
            w = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ (t[train] - tm))
            pred[test] = b @ w + tm
        r2 = 1 - np.sum((t - pred) ** 2) / np.sum((t - t.mean()) ** 2)
        best = max(best, r2)
    return float(best)


def radial_profile(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean of a cube around its centre in 1-cell shells: (radius in cells, mean)."""
    c = np.array(field.shape) // 2
    idx = np.indices(field.shape)
    r = np.sqrt(sum((idx[a] - c[a]) ** 2 for a in range(3)))
    shell = r.astype(int).ravel()
    sums = np.bincount(shell, weights=field.ravel())
    counts = np.bincount(shell)
    keep = counts > 0
    return np.arange(len(sums))[keep], sums[keep] / counts[keep]


def distance_to_tracer(raw: np.ndarray, region: Region, pad: int = 40) -> np.ndarray:
    """Distance in cells from each voxel of ``region`` to the nearest non-empty cell.

    ``raw`` is the un-normalized tracer field over the whole periodic box.
    The region is padded by ``pad`` cells on every side (wrapping around the
    box) before the distance transform, so distances up to ``pad`` are
    exact; larger ones are capped near ``pad``.
    """
    n = raw.shape[0]
    along = np.arange(region.start - pad, region.stop + pad) % n
    block = np.take(raw, along, axis=region.axis) > 0
    other = [a for a in range(3) if a != region.axis]
    block = np.pad(block, [(0, 0) if a == region.axis else (pad, pad) for a in range(3)],
                   mode="wrap")
    dist = ndimage.distance_transform_edt(~block).astype(np.float32)
    keep = [slice(pad, -pad)] * 3
    out = dist[tuple(keep)]
    assert out.shape[region.axis] == region.size and all(out.shape[a] == n for a in other)
    return out


def binned_scores(dist: np.ndarray, pred: np.ndarray, labels: np.ndarray,
                  edges: np.ndarray, n_classes: int = 4) -> dict:
    """Accuracy, per-class recall and voxel counts in bins of ``dist``."""
    which = np.digitize(dist.ravel(), edges) - 1
    ok = (which >= 0) & (which < len(edges) - 1)
    which, p, y = which[ok], pred.ravel()[ok], labels.ravel()[ok]
    nb = len(edges) - 1
    counts = np.bincount(which, minlength=nb)
    correct = np.bincount(which, weights=(p == y), minlength=nb)
    out = {"edges": edges, "counts": counts,
           "accuracy": np.divide(correct, counts, out=np.full(nb, np.nan), where=counts > 0)}
    for c in range(n_classes):
        m = y == c
        tot = np.bincount(which[m], minlength=nb)
        hit = np.bincount(which[m], weights=(p[m] == c), minlength=nb)
        out[f"recall_{c}"] = np.divide(hit, tot, out=np.full(nb, np.nan), where=tot > 0)
    return out


def entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy (bits) of the class probabilities along the last axis."""
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log2(p)).sum(-1)


def periodic_crops(x: np.ndarray, origins, size: int) -> np.ndarray:
    """Crops of side ``size`` from a periodic (n, n, n, ...) array, stacked."""
    n = x.shape[0]
    out = []
    for o in origins:
        ix = [np.arange(o[a], o[a] + size) % n for a in range(3)]
        out.append(x[np.ix_(*ix)])
    return np.stack(out)


def pooled(acts: np.ndarray) -> np.ndarray:
    """Per-crop mean and standard deviation of each channel: (b, ..., C) -> (b, 2C)."""
    flat = acts.reshape(acts.shape[0], -1, acts.shape[-1])
    return np.concatenate([flat.mean(1), flat.std(1)], axis=1)


# ---------------------------------------------------------------- needs TensorFlow

def block_names(model) -> list[str]:
    """Output layer of each U-Net block, from input to output side."""
    names = [layer.name for layer in model.layers]
    enc = sorted({n.split("_")[0] for n in names if n.startswith("enc") and n.endswith("relu2")})
    dec = sorted({n.split("_")[0] for n in names if n.startswith("dec") and n.endswith("relu2")},
                 reverse=True)
    return [f"{b}_relu2" for b in enc] + ["bottom_relu2"] + [f"{b}_relu2" for b in dec]


def model_input(model, x: np.ndarray, spacing: float | None = None):
    """A batch in the form the model takes: the array, or a dict with the spacing."""
    from .models import is_conditioned
    if is_conditioned(model):
        s = np.full((len(x), 1, 1, 1, 1), spacing, np.float32)
        return {"density": x.astype(np.float32), "spacing": s}
    return x.astype(np.float32)


def activations(model, x: np.ndarray, names: list[str], spacing: float | None = None,
                batch_size: int = 2, reduce=None) -> dict[str, np.ndarray]:
    """Outputs of the named layers for a batch of crops x (b, c, c, c, channels).

    ``reduce`` (e.g. :func:`pooled`) is applied to each batch's outputs as
    they come, so only the reduced features are kept in memory.
    """
    import keras
    probe = keras.Model(model.inputs, [model.get_layer(n).output for n in names])
    outs = {n: [] for n in names}
    for i in range(0, len(x), batch_size):
        res = probe.predict_on_batch(model_input(model, x[i:i + batch_size], spacing))
        for n, r in zip(names, res if isinstance(res, list) else [res]):
            r = np.asarray(r, np.float32)
            outs[n].append(reduce(r) if reduce else r)
    return {n: np.concatenate(v) for n, v in outs.items()}


def input_gradient(model, x: np.ndarray, spacing: float | None = None) -> np.ndarray:
    """|d log p(predicted class at the centre voxel) / d input|, averaged over crops.

    x is (b, c, c, c, channels); returns (c, c, c), summed over channels.
    """
    import tensorflow as tf

    from .models import is_conditioned
    conditioned = is_conditioned(model)
    c = x.shape[1] // 2
    total = 0.0
    for i in range(len(x)):
        inp = tf.convert_to_tensor(x[i:i + 1], tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(inp)
            feed = inp if not conditioned else {
                "density": inp, "spacing": tf.fill([1, 1, 1, 1, 1], float(spacing))}
            p = model(feed, training=False)[0, c, c, c]
            target = tf.math.log(tf.reduce_max(p) + 1e-12)
        g = tape.gradient(target, inp)[0].numpy()
        total = total + np.abs(g).sum(-1)
    return total / len(x)
