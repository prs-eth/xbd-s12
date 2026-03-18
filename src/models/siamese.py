import torch
import torch.nn as nn

from src.models.unet import UNetWithInputSkip
from src.models.utils import init_weights


class SiameseUnetLateFusion(nn.Module):

    def __init__(self, **kwargs):
        """
        Siamese U-Net with late fusion of two inputs.

        Args:
            kwargs: Arguments for UNetWithInputSkip
        """
        super().__init__()
        self.shared_unet = UNetWithInputSkip(**kwargs)

        # Final classification layer after concatenation
        final_channels = self.shared_unet.decoder_channels[-1] * 2
        out_channels = self.shared_unet.out_channels
        self.res = nn.Conv2d(final_channels, out_channels, kernel_size=1)

        if kwargs.get("verbose", False):
            print(f"SiameseUnetLateFusion created with final classification layer from {final_channels} to {out_channels} classes.")

        self.initialize_weights()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Forward pass through both U-Nets
        features1 = self.shared_unet(x1, return_features=True)
        features2 = self.shared_unet(x2, return_features=True)

        # Concatenate features from both U-Nets
        combined_features = torch.cat([features1, features2], dim=1)

        # Final classification
        output = self.res(combined_features)

        return output

    def initialize_weights(self) -> None:
        # Initialize only the final classification layer
        init_weights(self.res)
