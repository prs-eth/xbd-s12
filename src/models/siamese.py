import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from src.models.unet import UNetWithInputSkip
from src.models.utils import init_weights


class SiameseUnetLateFusion(nn.Module, PyTorchModelHubMixin):
    """
    Siamese U-Net with late fusion of two inputs (pre- and post-event).

    Both branches share weights (UNetWithInputSkip). Features from both
    branches are concatenated before the final classification layer.

    Supports push_to_hub / from_pretrained via PyTorchModelHubMixin.
    All __init__ kwargs are saved as config.json on the Hub.
    """

    def __init__(
        self,
        encoder_name="resnet34",
        encoder_depth=5,
        encoder_weights="imagenet",
        decoder_use_norm="batchnorm",
        decoder_channels=(256, 128, 64, 32, 16),
        decoder_attention_type=None,
        add_skip_connection=False,
        in_channels=14,
        out_channels=3,
        activation=None,
        verbose: bool = False,
        normalization_stats: dict = None,  # For Hugging Face Hub
    ):

        # NOTE: all arguments here are forwarded to UNetWithInputSkip AND saved as
        # config.json by the mixin. Keep this signature stable.
        super().__init__()

        self.shared_unet = UNetWithInputSkip(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_use_norm=decoder_use_norm,
            decoder_channels=list(decoder_channels),
            decoder_attention_type=decoder_attention_type,
            add_skip_connection=add_skip_connection,
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            verbose=verbose,
            normalization_stats=normalization_stats,
        )

        # After concatenating features from both branches
        final_channels = self.shared_unet.decoder_channels[-1] * 2
        out_channels = self.shared_unet.out_channels
        self.res = nn.Conv2d(final_channels, out_channels, kernel_size=1)

        if verbose:
            print(f"SiameseUnetLateFusion created with final classification layer from {final_channels} to {out_channels} classes.")

        self.initialize_weights()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        features1 = self.shared_unet(x1, return_features=True)
        features2 = self.shared_unet(x2, return_features=True)
        combined_features = torch.cat([features1, features2], dim=1)
        return self.res(combined_features)

    def initialize_weights(self) -> None:
        # Initialize only the final classification layer
        init_weights(self.res)
