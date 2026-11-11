"""Loss functions. Requires TensorFlow / Keras 3.

Both losses are differentiable in every term and serializable, so saved
models reload without extra bookkeeping.
"""

from __future__ import annotations

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="deepvoid")
class WeightedSCCE(keras.losses.Loss):
    """Sparse categorical cross-entropy with optional per-class weights."""

    def __init__(self, class_weights=None, name="weighted_scce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_weights = None if class_weights is None else [float(w) for w in class_weights]

    def call(self, y_true, y_pred):
        y_true = ops.cast(y_true, "int32")
        y_pred = ops.clip(ops.cast(y_pred, "float32"), 1e-7, 1.0)
        ce = -ops.log(ops.take_along_axis(y_pred, y_true[..., None], axis=-1))[..., 0]
        if self.class_weights is None:
            return ce
        return ce * ops.take(ops.convert_to_tensor(self.class_weights), y_true)

    def get_config(self):
        return {**super().get_config(), "class_weights": self.class_weights}


@keras.saving.register_keras_serializable(package="deepvoid")
class DiceSCCE(WeightedSCCE):
    """Weighted cross-entropy plus ``dice_weight`` times the soft Dice loss.

    The Dice term is 1 minus the mean soft Dice score over classes, computed
    over the whole batch, so every class counts equally however rare it is.
    """

    def __init__(self, class_weights=None, dice_weight=1.0, n_classes=4,
                 name="dice_scce", **kwargs):
        super().__init__(class_weights=class_weights, name=name, **kwargs)
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


def build_loss(cfg: dict, class_weights):
    name = cfg["name"]
    if name == "scce":
        return WeightedSCCE(class_weights)
    if name == "dice_scce":
        return DiceSCCE(class_weights, dice_weight=cfg["dice_weight"])
    raise ValueError(f"unknown loss {name!r}; use 'scce' or 'dice_scce'")
