"""Network architectures. Requires TensorFlow / Keras 3."""

from .film import FiLM, SpacingEmbedding
from .unet import copy_weights, is_conditioned, main_output, unet3d


def load_model(path, compile: bool = False):
    """Load a saved DeepVoid model.

    Imports the package's custom losses first so Keras can find them. By
    default the model is loaded for prediction only (``compile=False``);
    pass ``compile=True`` to restore the loss and optimizer as well.
    """
    import keras

    from .. import losses  # noqa: F401  registers the custom losses with Keras

    return keras.models.load_model(path, compile=compile)


__all__ = ["unet3d", "load_model", "copy_weights", "is_conditioned", "main_output", "FiLM",
           "SpacingEmbedding"]
