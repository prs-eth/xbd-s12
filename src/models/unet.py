import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from src.models.utils import init_weights


class UNetWithInputSkip(nn.Module, PyTorchModelHubMixin):
    """SMP U-Net extended with an extra processing block before the segmentation head.

    Supports push_to_hub/from_pretrained via PyTorchModelHubMixin.

    Note:
        This class was originally designed to include a skip connection from the input
        to a final processing block (hence the name). The skip connection did not bring
        significant improvements and was removed, but the extra ``final_block`` was
        inadvertently kept. The model is therefore equivalent to a standard SMP U-Net
        followed by an additional two-layer convolutional block before the segmentation
        head. This is believed to be harmless to performance.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str = "imagenet",
        decoder_use_norm: str = "batchnorm",
        decoder_channels: list[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: str = None,
        add_skip_connection: bool = False,
        in_channels: int = 3,
        out_channels: int = 1,
        activation: str = None,
        verbose: bool = False,
        normalization_stats: dict = None,  # For Hugging Face Hub
    ):
        """
        Standard SMP U-Net with additional skip connection from input if desired.

        Args:
            encoder_name (str): Name of the encoder to use.
            encoder_depth (int): Depth of the encoder.
            encoder_weights (str): Pretrained weights to use for the encoder.
            decoder_use_norm (str): Normalization to use in the decoder.
            decoder_channels (list[int]): Number of channels in each decoder block.
            decoder_attention_type (str): Attention type to use in the decoder.
            add_skip_connection (bool): Whether to add skip connection from input to final block.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (str): Activation function to use in the final layer.
            verbose (bool): Whether to print model details.
            normalization_stats (dict): Normalization stats to include in the model config for Hugging Face Hub.
        """
        super().__init__()

        # Adjust decoder_channels based on encoder_depth
        if encoder_depth < len(decoder_channels):
            decoder_channels = decoder_channels[:encoder_depth]
            if verbose:
                print(f"Adjusted decoder_channels to {decoder_channels} based on encoder_depth={encoder_depth}")
        elif encoder_depth > len(decoder_channels):
            raise ValueError(
                f"encoder_depth ({encoder_depth}) cannot be greater than "
                f"length of decoder_channels ({len(decoder_channels)}). "
                f"Please provide more decoder channels."
            )
        self.add_skip_connection = add_skip_connection
        self.decoder_channels = decoder_channels
        self.out_channels = out_channels
        self.normalization_stats = normalization_stats  # For Hugging Face Hub

        # Create standard SMP U-Net
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_use_norm=decoder_use_norm,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels,
            classes=decoder_channels[-1],  # Output features instead of final classes
            activation=None,  # No activation yet
        )

        if verbose:
            print(f"UNet with encoder {encoder_name}, depth {encoder_depth}, in_channels {in_channels}, out_channels {decoder_channels[-1]} created")
        if encoder_weights is not None and verbose:
            print(f"Using encoder weights: {encoder_weights}")

        if self.add_skip_connection:
            if verbose:
                print("Adding skip connection from input to final block in UNet.")
            # Skip connection from input
            self.skip_input = nn.Sequential(
                nn.Conv2d(in_channels, decoder_channels[-1], kernel_size=1), nn.BatchNorm2d(decoder_channels[-1]), nn.ReLU(inplace=True)
            )

        # Additional processing after concatenation (SHOULD HAVE BEEN AVOIDED SINCE NO SKIP...)
        input_final_block = decoder_channels[-1] * 2 if add_skip_connection else decoder_channels[-1]
        self.final_block = nn.Sequential(
            nn.Conv2d(input_final_block, decoder_channels[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[-1]),
            nn.ReLU(inplace=True),
        )

        # Final segmentation head
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)
        if verbose:
            print(f"Final segmentation head from {decoder_channels[-1]} to {out_channels} classes created.")

        # Optional activation
        self.activation = self._get_activation(activation)

        # Initialize weights of new layers
        self.initialize_weights()

    def _get_activation(self, activation):
        if activation is None or activation == "identity":
            return nn.Identity()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "softmax":
            return nn.Softmax(dim=1)
        else:
            raise ValueError(f"Activation {activation} is not supported")

    def initialize_weights(self):
        # Don't initialize the UNet, just the new layers
        if self.add_skip_connection:
            init_weights(self.skip_input)
        init_weights(self.final_block)
        init_weights(self.segmentation_head)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        # Get U-Net features
        unet_output = self.unet(x)

        # Skip connection from input
        if self.add_skip_connection:
            x_skip = self.skip_input(x)

            # Concatenate and process
            combined = torch.cat([unet_output, x_skip], dim=1)
            features = self.final_block(combined)
        else:
            # (SHOULD HAVE BEEN AVOIDED IF NO SKIP...)
            features = self.final_block(unet_output)

        if return_features:
            return features

        # Final segmentation
        output = self.segmentation_head(features)

        # Apply activation if specified
        output = self.activation(output)

        return output
