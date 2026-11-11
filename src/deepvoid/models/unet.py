"""3D U-Net for voxel-wise classification."""

from __future__ import annotations

import keras
from keras import layers


def _conv_block(x, filters: int, batch_norm: bool, name: str):
    for i in (1, 2):
        x = layers.Conv3D(filters, 3, padding="same", use_bias=not batch_norm,
                          name=f"{name}_conv{i}")(x)
        if batch_norm:
            x = layers.BatchNormalization(momentum=0.9, name=f"{name}_bn{i}")(x)
        x = layers.Activation("relu", name=f"{name}_relu{i}")(x)
    return x


def unet3d(depth: int = 4, base_filters: int = 16, n_classes: int = 4,
           in_channels: int = 1, batch_norm: bool = True, dropout: float = 0.0,
           name: str = "unet3d") -> keras.Model:
    """Build a 3D U-Net.

    The input side length is left open, so a model trained on 64^3 crops can
    predict on larger cubes, as long as the side is divisible by
    2^(depth - 1). Layers are named by block (``enc0``, ``enc1``, ...,
    ``bottom``, ``dec...``) so they can be frozen by name for curriculum
    training. The output is a float32 softmax, which keeps it stable under
    mixed precision. Batch-norm momentum is 0.9 rather than Keras's 0.99 so
    the running statistics used at inference settle within a few hundred
    steps.
    """
    inputs = keras.Input(shape=(None, None, None, in_channels), name="density")
    x, skips = inputs, []
    for level in range(depth - 1):
        x = _conv_block(x, base_filters * 2 ** level, batch_norm, f"enc{level}")
        skips.append(x)
        x = layers.MaxPooling3D(2, name=f"enc{level}_pool")(x)

    x = _conv_block(x, base_filters * 2 ** (depth - 1), batch_norm, "bottom")
    if dropout:
        x = layers.Dropout(dropout, name="bottom_dropout")(x)

    for level in reversed(range(depth - 1)):
        x = layers.Conv3DTranspose(base_filters * 2 ** level, 2, strides=2,
                                   name=f"dec{level}_up")(x)
        x = layers.Concatenate(name=f"dec{level}_concat")([x, skips[level]])
        x = _conv_block(x, base_filters * 2 ** level, batch_norm, f"dec{level}")

    logits = layers.Conv3D(n_classes, 1, name="logits")(x)
    probs = layers.Activation("softmax", dtype="float32", name="probs")(logits)
    return keras.Model(inputs, probs, name=name)


def freeze_encoder(model: keras.Model, levels: int) -> None:
    """Freeze the first ``levels`` encoder blocks (all their layers, by name)."""
    prefixes = tuple(f"enc{i}_" for i in range(levels))
    for layer in model.layers:
        if layer.name.startswith(prefixes):
            layer.trainable = False
