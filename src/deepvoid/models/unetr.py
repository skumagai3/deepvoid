"""3D UNETR: a vision transformer encoder with a convolutional decoder.

Follows Hatamizadeh et al. 2022 (arXiv:2103.10504). The crop is cut into
cubic patches, each patch becomes one token (a vector of ``dim`` numbers),
and a stack of transformer blocks lets every token attend to every other
token in the crop. Token grids taken from several depths of the stack are
upsampled and merged, U-Net style, into a decoder that ends at full
resolution with one class score per voxel per class.

Differences from the paper (see docs/training.md):

- Position embedding: fixed sine/cosine per axis (Vaswani et al. 2017) in
  place of the paper's learned embedding. It is defined for any token grid,
  so the 64^3 validation crops and the training crops use the same model
  unchanged. Prediction crops should not exceed the training crop, or the
  model meets token positions it never saw in training.
- Stem: optionally (``stem="conv"``) a stack of stride-2 convolutions in
  place of the single patch convolution (Xiao et al. 2021, see
  :func:`_conv_stem`).
- Size: by default 8^3 patches, 9 blocks of width 240 (about 6M weights in
  the transformer) in place of 16^3 patches and 12 blocks of width 768
  (92.6M weights in the whole model).
- Decoder: the same convolution blocks as ``unet3d`` (two 3x3x3
  convolutions, each with batch normalization and ReLU).
"""

from __future__ import annotations

import math

import keras
from keras import layers, ops

from .unet import _conv_block


@keras.saving.register_keras_serializable(package="deepvoid")
class SinePosition3D(layers.Layer):
    """Add a fixed sine/cosine position embedding to a grid of tokens.

    Input and output shape: (batch, n1, n2, n3, dim). The ``dim`` channels
    are split into three equal blocks, one per axis. Along an axis, the
    token at index n gets sin(n w_k) and cos(n w_k) for the frequencies
    w_k = 10000^(-k / K), k = 0 ... K - 1, with K = dim / 6. The embedding
    has no weights and is computed from the grid shape at each call, so it
    works for any grid size.

    The dot product of two tokens' embeddings, sum_k cos((n - n') w_k) per
    axis, depends only on their offset along each axis.
    """

    def __init__(self, base: float = 10000.0, **kwargs):
        super().__init__(**kwargs)
        self.base = base

    def build(self, input_shape):
        dim = input_shape[-1]
        if dim % 6:
            raise ValueError(f"the token width must be divisible by 6, got {dim}")
        self.n_freq = dim // 6

    def _axis_embedding(self, n):
        k = ops.arange(self.n_freq, dtype="float32")
        freqs = ops.power(self.base, -k / self.n_freq)                 # (K,)
        angles = ops.arange(n, dtype="float32")[:, None] * freqs[None]  # (n, K)
        return ops.concatenate([ops.sin(angles), ops.cos(angles)], axis=-1)  # (n, 2K)

    def call(self, x):
        n1, n2, n3 = ops.shape(x)[1], ops.shape(x)[2], ops.shape(x)[3]
        e1 = self._axis_embedding(n1)[:, None, None, :]
        e2 = self._axis_embedding(n2)[None, :, None, :]
        e3 = self._axis_embedding(n3)[None, None, :, :]
        zeros = ops.zeros((n1, n2, n3, 1), dtype="float32")
        emb = ops.concatenate([e1 + zeros, e2 + zeros, e3 + zeros], axis=-1)
        return x + ops.cast(emb[None], x.dtype)

    def get_config(self):
        return {**super().get_config(), "base": self.base}


@keras.saving.register_keras_serializable(package="deepvoid")
class TransformerBlock(layers.Layer):
    """One pre-norm transformer block acting on a grid of tokens.

    Input and output shape: (batch, n1, n2, n3, dim). The grid is flattened
    to a list of n1 n2 n3 tokens, then

        x = x + attention(layer_norm(x))
        x = x + mlp(layer_norm(x))

    where attention is multi-head self-attention over all tokens in the crop
    and mlp is two dense layers (dim -> mlp_ratio * dim -> dim) with a GELU
    between them. The result is reshaped back to the grid.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4, **kwargs):
        super().__init__(**kwargs)
        if dim % heads:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        self.dim, self.heads, self.mlp_ratio = dim, heads, mlp_ratio
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.attention = layers.MultiHeadAttention(heads, dim // heads, name="attention")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        self.mlp_in = layers.Dense(mlp_ratio * dim, activation="gelu", name="mlp_in")
        self.mlp_out = layers.Dense(dim, name="mlp_out")

    def build(self, input_shape):
        seq = (input_shape[0], None, self.dim)
        self.norm1.build(seq)
        self.attention.build(seq, seq)
        self.norm2.build(seq)
        self.mlp_in.build(seq)
        self.mlp_out.build((input_shape[0], None, self.mlp_ratio * self.dim))

    def call(self, x):
        grid = ops.shape(x)
        x = ops.reshape(x, (grid[0], -1, self.dim))
        h = self.norm1(x)
        x = x + self.attention(h, h)
        x = x + self.mlp_out(self.mlp_in(self.norm2(x)))
        return ops.reshape(x, grid)

    def get_config(self):
        return {**super().get_config(), "dim": self.dim, "heads": self.heads,
                "mlp_ratio": self.mlp_ratio}


def _up_block(x, filters: int, batch_norm: bool, bn_momentum: float, name: str):
    """Double the resolution: a 2x2x2 transposed convolution, then a 3x3x3
    convolution with batch normalization and ReLU (as in UNETR's skip paths)."""
    x = layers.Conv3DTranspose(filters, 2, strides=2, name=f"{name}_up")(x)
    x = layers.Conv3D(filters, 3, padding="same", use_bias=not batch_norm,
                      name=f"{name}_conv")(x)
    if batch_norm:
        x = layers.BatchNormalization(momentum=bn_momentum, name=f"{name}_bn")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


def _conv_stem(x, patch: int, dim: int, batch_norm: bool, bn_momentum: float):
    """Tokens from a stack of stride-2 convolutions, after Xiao et al. 2021
    ("Early Convolutions Help Transformers See Better", arXiv:2106.14881).

    With patch = 2^m: m 3x3x3 convolutions of stride 2, each followed by
    batch normalization and ReLU, with channels doubling up to ``dim``
    (dim/4, dim/2, dim for patch 8), then a 1x1x1 convolution to ``dim``
    with no normalization or activation, as in the paper. The token grid is
    the same as the single patch convolution's (1/patch resolution), but
    neighbouring patches now overlap in what each token sees, and the
    reduction to one token is spread over m nonlinear steps.

    The paper removes one transformer block to keep the computation equal;
    that is left to ``blocks`` here.
    """
    m = int(round(math.log2(patch)))
    if dim % 2 ** (m - 1):
        raise ValueError(f"dim ({dim}) must be divisible by {2 ** (m - 1)} for the conv stem")
    for i in range(m):
        filters = dim // 2 ** (m - 1 - i)
        x = layers.Conv3D(filters, 3, strides=2, padding="same", use_bias=not batch_norm,
                          name=f"stem{i}_conv")(x)
        if batch_norm:
            x = layers.BatchNormalization(momentum=bn_momentum, name=f"stem{i}_bn")(x)
        x = layers.Activation("relu", name=f"stem{i}_relu")(x)
    return layers.Conv3D(dim, 1, name="patch_embed")(x)


def tap_blocks(patch: int, blocks: int) -> dict[int, int]:
    """Which transformer block feeds each decoder level, as {level: block}.

    With patch = 2^m, level j (1/2^j resolution, j = 1 ... m) takes block
    round(j * blocks / m); level m is the bottleneck and takes the last block.
    """
    m = int(round(math.log2(patch)))
    return {level: round(level * blocks / m) for level in range(1, m + 1)}


def unetr3d(patch: int = 8, dim: int = 240, blocks: int = 9, heads: int = 8,
            mlp_ratio: int = 4, stem: str = "patch", base_filters: int = 16, n_classes: int = 4,
            in_channels: int = 1, batch_norm: bool = True, bn_momentum: float = 0.99,
            dropout: float = 0.0, name: str = "unetr3d") -> keras.Model:
    """Build a 3D UNETR.

    ``patch`` must be a power of two, and the input side a multiple of it.
    With patch = 2^m, the decoder has m + 1 levels, at full resolution and at
    1/2, 1/4, ..., 1/patch of it, with ``base_filters`` * 2^level channels
    (as in ``unet3d`` of depth m + 1).

    The decoder is fed by the input itself (through one convolution block,
    at full resolution) and by the token grids after transformer blocks
    round(level * blocks / m) for level = 1 ... m. Block ``blocks`` (the
    last) is the bottleneck, at 1/patch; earlier blocks feed finer levels,
    each grid brought up to its level by repeated :func:`_up_block`. With
    patch 16 and 12 blocks this picks blocks 3, 6, 9 and 12, as in the
    paper.

    ``stem`` is "patch" (one convolution with kernel and stride ``patch``,
    as in the paper) or "conv" (:func:`_conv_stem`).

    Layers are named ``stem{i}_...`` (conv stem only), ``patch_embed``, ``position``, ``block{i}`` (from 1),
    ``skip{level}_...``, ``dec{level}_...``, ``logits`` and ``probs``. The
    output is a float32 softmax, as in ``unet3d``.
    """
    m = int(round(math.log2(patch)))
    if 2 ** m != patch or m < 1:
        raise ValueError(f"patch must be a power of two >= 2, got {patch}")
    if blocks < m:
        raise ValueError(f"need at least {m} transformer blocks for patch {patch}")
    block = dict(batch_norm=batch_norm, bn_momentum=bn_momentum)
    taps = tap_blocks(patch, blocks)

    inputs = keras.Input(shape=(None, None, None, in_channels), name="density")

    # Encoder: patches to tokens, position, transformer blocks.
    if stem == "patch":
        x = layers.Conv3D(dim, patch, strides=patch, name="patch_embed")(inputs)
    elif stem == "conv":
        x = _conv_stem(inputs, patch, dim, batch_norm, bn_momentum)
    else:
        raise ValueError(f"stem must be 'patch' or 'conv', got {stem!r}")
    x = SinePosition3D(name="position")(x)
    tapped = {}
    for i in range(1, blocks + 1):
        x = TransformerBlock(dim, heads, mlp_ratio, name=f"block{i}")(x)
        if i in taps.values():
            tapped[i] = x

    # Skip paths: each tapped token grid upsampled to its decoder level.
    skips = {0: _conv_block(inputs, base_filters, name="skip0", **block)}
    for level in range(1, m):
        s = tapped[taps[level]]
        for step in range(m - level):
            s = _up_block(s, base_filters * 2 ** level, name=f"skip{level}_{step}", **block)
        skips[level] = s

    # Decoder: from the bottleneck (last block, at 1/patch) up to full resolution.
    x = tapped[taps[m]]
    if dropout:
        x = layers.Dropout(dropout, name="bottom_dropout")(x)
    for level in reversed(range(m)):
        x = layers.Conv3DTranspose(base_filters * 2 ** level, 2, strides=2,
                                   name=f"dec{level}_up")(x)
        x = layers.Concatenate(name=f"dec{level}_concat")([x, skips[level]])
        x = _conv_block(x, base_filters * 2 ** level, name=f"dec{level}", **block)

    logits = layers.Conv3D(n_classes, 1, name="logits")(x)
    probs = layers.Activation("softmax", dtype="float32", name="probs")(logits)
    return keras.Model(inputs, probs, name=name)
