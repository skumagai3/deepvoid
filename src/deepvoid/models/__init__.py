"""Network architectures. Requires TensorFlow / Keras 3."""

from .film import FiLM, SpacingEmbedding
from .unet import copy_weights, is_conditioned, main_output, unet3d
from .unetr import SinePosition3D, TransformerBlock, unetr3d


def build_model(model_cfg: dict, in_channels: int = 1):
    """Build the network described by a config's ``model`` section."""
    cfg = dict(model_cfg)
    arch, vit = cfg.pop("arch", "unet"), cfg.pop("vit", None)
    if arch == "unet":
        return unet3d(**cfg, in_channels=in_channels)
    if arch == "unetr":
        return unetr3d(**vit, base_filters=cfg["base_filters"], batch_norm=cfg["batch_norm"],
                       bn_momentum=cfg["bn_momentum"], dropout=cfg["dropout"],
                       in_channels=in_channels)
    raise ValueError(f"unknown model.arch {arch!r}")


def load_model(path, compile: bool = False):
    """Load a saved DeepVoid model.

    Imports the package's custom losses first so Keras can find them. By
    default the model is loaded for prediction only (``compile=False``);
    pass ``compile=True`` to restore the loss and optimizer as well.
    """
    import keras

    from .. import losses  # noqa: F401  registers the custom losses with Keras

    return keras.models.load_model(path, compile=compile)


__all__ = ["unet3d", "unetr3d", "build_model", "load_model", "copy_weights", "is_conditioned", "main_output", "FiLM",
           "SpacingEmbedding", "SinePosition3D", "TransformerBlock"]
