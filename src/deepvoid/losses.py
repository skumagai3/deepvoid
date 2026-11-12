"""Loss functions. Requires TensorFlow / Keras 3.

All losses are differentiable in every term and serializable, so saved
models reload without extra bookkeeping.

- :class:`WeightedSCCE`: cross-entropy over the four classes, with optional
  per-class weights, plus optionally ``void_weight`` times a void-vs-rest
  term: the binary cross-entropy of the void score s_v against the void
  label, -[1(void) log s_v + (1 - 1(void)) log(1 - s_v)]. That term trains
  the void score directly for the two-class decision voids are judged on.
- :class:`DiceSCCE`: the same, plus a soft Dice term.
- :class:`DeepSupervisionLoss`: cross-entropy for an extra output at a
  coarser resolution, against the class fractions in each block of the
  full-resolution labels (see ``models.unet``).
"""

from __future__ import annotations

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="deepvoid")
class WeightedSCCE(keras.losses.Loss):
    """Sparse categorical cross-entropy with optional per-class weights.

    With ``void_weight`` > 0, adds that multiple of the void-vs-rest binary
    cross-entropy of the void score (class 0) at every voxel.
    """

    def __init__(self, class_weights=None, void_weight=0.0, name="weighted_scce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_weights = None if class_weights is None else [float(w) for w in class_weights]
        self.void_weight = float(void_weight)

    def call(self, y_true, y_pred):
        y_true = ops.cast(y_true, "int32")
        y_pred = ops.clip(ops.cast(y_pred, "float32"), 1e-7, 1.0 - 1e-7)
        ce = -ops.log(ops.take_along_axis(y_pred, y_true[..., None], axis=-1))[..., 0]
        if self.class_weights is not None:
            ce = ce * ops.take(ops.convert_to_tensor(self.class_weights), y_true)
        if self.void_weight:
            is_void = ops.cast(ops.equal(y_true, 0), "float32")
            s_void = y_pred[..., 0]
            bce = -(is_void * ops.log(s_void) + (1.0 - is_void) * ops.log(1.0 - s_void))
            ce = ce + self.void_weight * bce
        return ce

    def get_config(self):
        return {**super().get_config(), "class_weights": self.class_weights,
                "void_weight": self.void_weight}


@keras.saving.register_keras_serializable(package="deepvoid")
class DiceSCCE(WeightedSCCE):
    """Weighted cross-entropy plus ``dice_weight`` times the soft Dice loss.

    The Dice term is 1 minus the mean soft Dice score over classes, computed
    over the whole batch, so every class counts equally however rare it is.
    """

    def __init__(self, class_weights=None, dice_weight=1.0, n_classes=4, void_weight=0.0,
                 name="dice_scce", **kwargs):
        super().__init__(class_weights=class_weights, void_weight=void_weight, name=name,
                         **kwargs)
        self.dice_weight = float(dice_weight)
        self.n_classes = int(n_classes)

    def call(self, y_true, y_pred):
        ce = super().call(y_true, y_pred)
        p = ops.cast(y_pred, "float32")
        t = ops.one_hot(ops.cast(y_true, "int32"), self.n_classes)
        axes = tuple(range(len(p.shape) - 1))
        intersection = ops.sum(p * t, axis=axes)
        total = ops.sum(p, axis=axes) + ops.sum(t, axis=axes)
        dice = (2.0 * intersection + 1.0) / (total + 1.0)
        return ce + self.dice_weight * (1.0 - ops.mean(dice))

    def get_config(self):
        return {**super().get_config(), "dice_weight": self.dice_weight,
                "n_classes": self.n_classes}


@keras.saving.register_keras_serializable(package="deepvoid")
class DeepSupervisionLoss(keras.losses.Loss):
    """Cross-entropy for an output ``factor`` times coarser than the labels.

    The target for each coarse cell is the fraction of each class among the
    factor^3 full-resolution voxels it covers (the one-hot labels averaged
    over the block), so blocks that straddle a boundary keep both classes
    rather than being forced to one. Per-class weights, if given, weight each
    cell by sum_c fraction_c * weight_c.
    """

    def __init__(self, factor: int, class_weights=None, n_classes=4,
                 name="deep_supervision", **kwargs):
        super().__init__(name=name, **kwargs)
        self.factor = int(factor)
        self.class_weights = None if class_weights is None else [float(w) for w in class_weights]
        self.n_classes = int(n_classes)

    def call(self, y_true, y_pred):
        t = ops.one_hot(ops.cast(y_true, "int32"), self.n_classes)
        t = ops.average_pool(t, pool_size=self.factor, strides=self.factor, padding="valid")
        p = ops.clip(ops.cast(y_pred, "float32"), 1e-7, 1.0)
        ce = -ops.sum(t * ops.log(p), axis=-1)
        if self.class_weights is not None:
            ce = ce * ops.sum(t * ops.convert_to_tensor(self.class_weights), axis=-1)
        return ce

    def get_config(self):
        return {**super().get_config(), "factor": self.factor,
                "class_weights": self.class_weights, "n_classes": self.n_classes}


def build_loss(cfg: dict, class_weights):
    name = cfg["name"]
    void_weight = cfg.get("void_weight", 0.0)
    if name == "scce":
        return WeightedSCCE(class_weights, void_weight=void_weight)
    if name == "dice_scce":
        return DiceSCCE(class_weights, dice_weight=cfg["dice_weight"], void_weight=void_weight)
    raise ValueError(f"unknown loss {name!r}; use 'scce' or 'dice_scce'")


def deep_supervision_weights(depth: int) -> list[float]:
    """Loss weights for the outputs at factors 1, 2, ..., 2^(depth-1): 1, 1/2, ... normalized."""
    w = [2.0 ** -level for level in range(depth)]
    return [x / sum(w) for x in w]
