"""Conditioning the U-Net on tracer spacing (FiLM).

The network is told the mean tracer spacing lambda = n^(-1/3) of its input,
which is known for a volume-limited sample. A small network turns
log(lambda) into an embedding, and in every convolution block, after
normalization, each feature map is scaled and shifted by amounts computed
from that embedding (feature-wise linear modulation, Perez et al. 2018):

    x -> x * (1 + gamma(e)) + beta(e)

The layers producing gamma and beta start at zero, so an untrained
conditioned model behaves exactly like the plain U-Net, and weights copied
from a plain model give the same predictions until training moves them.

The spacing enters as a 5D tensor (batch, 1, 1, 1, 1) that broadcasts over
the grid. A spatially varying spacing (a flux-limited survey, a mask) would
enter as (batch, x, y, z, 1) instead; the layers here already work
per-voxel, so that case needs only the map pooled to each block's
resolution.

Requires TensorFlow / Keras 3.
"""

from __future__ import annotations

import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="deepvoid")
class SpacingEmbedding(layers.Layer):
    """log(lambda) -> embedding of size ``units``, with optional training noise.

    ``noise`` is the standard deviation of a random factor exp(N(0, noise))
    applied to lambda during training only, so the model cannot depend on
    the exact value it was given; 0.1 is about a 10% scatter.
    """

    def __init__(self, units: int = 32, noise: float = 0.0, seed: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.units, self.noise, self.seed = units, noise, seed
        self.dense1 = layers.Dense(units, activation="silu", name="dense1")
        self.dense2 = layers.Dense(units, activation="silu", name="dense2")
        self.seed_generator = keras.random.SeedGenerator(seed)

    def build(self, input_shape):
        self.dense1.build(input_shape)
        self.dense2.build((*input_shape[:-1], self.units))

    def call(self, spacing, training=False):
        log_lam = ops.log(ops.maximum(ops.cast(spacing, "float32"), 1e-3))
        if training and self.noise > 0:
            log_lam = log_lam + self.noise * keras.random.normal(
                ops.shape(log_lam), seed=self.seed_generator)
        return self.dense2(self.dense1(ops.cast(log_lam, self.compute_dtype)))

    def get_config(self):
        return {**super().get_config(), "units": self.units, "noise": self.noise,
                "seed": self.seed}


@keras.saving.register_keras_serializable(package="deepvoid")
class FiLM(layers.Layer):
    """Scale and shift each channel of ``x`` by amounts computed from an embedding."""

    def __init__(self, channels: int, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.proj = layers.Dense(2 * channels, kernel_initializer="zeros",
                                 bias_initializer="zeros", name="proj")

    def build(self, input_shapes):
        self.proj.build(input_shapes[1])

    def call(self, inputs):
        x, embedding = inputs
        gamma, beta = ops.split(ops.cast(self.proj(embedding), x.dtype), 2, axis=-1)
        return x * (1.0 + gamma) + beta

    def get_config(self):
        return {**super().get_config(), "channels": self.channels}
