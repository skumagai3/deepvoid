"""Loss functions. Requires TensorFlow / Keras 3.

Every loss here is differentiable in all of its terms and serializable, so
saved models reload without custom-object bookkeeping.
"""

from __future__ import annotations

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="deepvoid")
class WeightedSCCE(keras.losses.Loss):
    """Sparse categorical cross-entropy with per-class weights.

    With no weights this is plain SCCE. Weights inversely proportional to
    class frequency counter the dominance of void and wall voxels.
    """

    def __init__(self, class_weights=None, name="weighted_scce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_weights = None if class_weights is None else list(map(float, class_weights))

    def call(self, y_true, y_pred):
        y_true = ops.cast(y_true, "int32")
        y_pred = ops.clip(y_pred, 1e-7, 1.0)
        ce = -ops.log(ops.take_along_axis(y_pred, y_true[..., None], axis=-1))[..., 0]
        if self.class_weights is None:
            return ce
        w = ops.take(ops.convert_to_tensor(self.class_weights), y_true)
        return ce * w

    def get_config(self):
        return {**super().get_config(), "class_weights": self.class_weights}
