"""3D U-Net for voxel-wise classification."""

from __future__ import annotations

import keras
from keras import layers

from .film import FiLM, SpacingEmbedding


def _conv_block(x, filters: int, batch_norm: bool, bn_momentum: float, name: str,
                embedding=None):
    for i in (1, 2):
        x = layers.Conv3D(filters, 3, padding="same", use_bias=not batch_norm,
                          name=f"{name}_conv{i}")(x)
        if batch_norm:
            x = layers.BatchNormalization(momentum=bn_momentum, name=f"{name}_bn{i}")(x)
        if embedding is not None:
            x = FiLM(filters, name=f"{name}_film{i}")([x, embedding])
        x = layers.Activation("relu", name=f"{name}_relu{i}")(x)
    return x


def unet3d(depth: int = 4, base_filters: int = 16, n_classes: int = 4,
           in_channels: int = 1, batch_norm: bool = True, bn_momentum: float = 0.99,
           dropout: float = 0.0, conditioning: dict | None = None,
           deep_supervision: bool = False, name: str = "unet3d") -> keras.Model:
    """Build a 3D U-Net.

    The input side length is left open, so a model trained on 64^3 crops can
    predict on larger cubes, as long as the side is divisible by
    2^(depth - 1). Layers are named by block (``enc0``, ``enc1``, ...,
    ``bottom``, ``dec...``) so they can be matched or frozen by name. The
    output is a float32 softmax, which keeps it stable under mixed precision.

    With ``conditioning`` (a dict of :class:`SpacingEmbedding` settings,
    e.g. ``{"units": 32, "noise": 0.1}``), the model takes a second input,
    ``spacing``, of shape (batch, 1, 1, 1, 1) holding the tracer spacing in
    h^-1 Mpc, and every convolution is followed by a FiLM layer (see
    ``film.py``). Inputs are then passed as a dict with keys ``density``
    and ``spacing``.

    With ``deep_supervision``, the model has one extra output per level
    below full resolution: after each decoder block at 1/2, 1/4, ... and
    after the bottleneck, a 1x1x1 convolution and softmax give class scores
    at that resolution (outputs ``ds1`` ... ``ds{depth-1}``, at 2^level times
    coarser). The outputs are then a list, full resolution first; these
    extra outputs are only for training (see ``losses.DeepSupervisionLoss``),
    and :func:`main_output` gives the model with the full-resolution output
    alone, which is what gets saved for prediction.
    """
    inputs = keras.Input(shape=(None, None, None, in_channels), name="density")
    emb = None
    model_inputs = inputs
    if conditioning is not None:
        spacing = keras.Input(shape=(None, None, None, 1), name="spacing")
        emb = SpacingEmbedding(**conditioning, name="spacing_embedding")(spacing)
        model_inputs = {"density": inputs, "spacing": spacing}
    block = dict(batch_norm=batch_norm, bn_momentum=bn_momentum, embedding=emb)
    x, skips = inputs, []
    for level in range(depth - 1):
        x = _conv_block(x, base_filters * 2 ** level, name=f"enc{level}", **block)
        skips.append(x)
        x = layers.MaxPooling3D(2, name=f"enc{level}_pool")(x)

    x = _conv_block(x, base_filters * 2 ** (depth - 1), name="bottom", **block)
    if dropout:
        x = layers.Dropout(dropout, name="bottom_dropout")(x)
    coarse = {depth - 1: x}

    for level in reversed(range(depth - 1)):
        x = layers.Conv3DTranspose(base_filters * 2 ** level, 2, strides=2,
                                   name=f"dec{level}_up")(x)
        x = layers.Concatenate(name=f"dec{level}_concat")([x, skips[level]])
        x = _conv_block(x, base_filters * 2 ** level, name=f"dec{level}", **block)
        if level > 0:
            coarse[level] = x

    logits = layers.Conv3D(n_classes, 1, name="logits")(x)
    probs = layers.Activation("softmax", dtype="float32", name="probs")(logits)
    if not deep_supervision:
        return keras.Model(model_inputs, probs, name=name)
    heads = []
    for level in range(1, depth):
        h = layers.Conv3D(n_classes, 1, name=f"ds{level}_logits")(coarse[level])
        heads.append(layers.Activation("softmax", dtype="float32", name=f"ds{level}")(h))
    return keras.Model(model_inputs, [probs] + heads, name=name)


def main_output(model: keras.Model) -> keras.Model:
    """The model with only its full-resolution output (shares all weights)."""
    if len(model.outputs) == 1:
        return model
    return keras.Model(model.input, model.outputs[0], name=model.name)


def is_conditioned(model: keras.Model) -> bool:
    """True if the model takes a ``spacing`` input."""
    return any(layer.name == "spacing_embedding" for layer in model.layers)


def copy_weights(source: keras.Model, target: keras.Model) -> tuple[list[str], list[str]]:
    """Copy weights layer by layer, matching layers by name and weight shapes.

    Used to start a run from a trained model, including a plain model into a
    conditioned one (the FiLM layers, absent from the source, keep their
    identity initialization). Returns the names of the layers copied and of
    the target layers with weights that were left as they were.
    """
    by_name = {layer.name: layer for layer in source.layers}
    copied, skipped = [], []
    for layer in target.layers:
        weights = layer.get_weights()
        if not weights:
            continue
        old = by_name.get(layer.name)
        old_weights = old.get_weights() if old is not None else []
        if [w.shape for w in old_weights] == [w.shape for w in weights]:
            layer.set_weights(old_weights)
            copied.append(layer.name)
        else:
            skipped.append(layer.name)
    return copied, skipped


def freeze_encoder(model: keras.Model, levels: int) -> None:
    """Freeze the first ``levels`` encoder blocks (all their layers, by name)."""
    prefixes = tuple(f"enc{i}_" for i in range(levels))
    for layer in model.layers:
        if layer.name.startswith(prefixes):
            layer.trainable = False
