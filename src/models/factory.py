import torch.nn as nn

from src.models.siamese import SiameseUnetLateFusion
from src.models.unet import UNetWithInputSkip


def model_factory(model_name: str, **model_kwargs) -> nn.Module:
    """
    Factory function to create a model instance.

    Args:
        model_name: The name of the model to create.
        **model_kwargs: Additional keyword arguments to pass to the model constructor.

    Returns:
        nn.Module: The created model instance.
    """

    if model_name == "unet":
        model = UNetWithInputSkip(**model_kwargs)

    elif model_name == "siamese":
        model = SiameseUnetLateFusion(**model_kwargs)

    else:
        raise ValueError(f"Model {model_name} not recognized.")

    return model
