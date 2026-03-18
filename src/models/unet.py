import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from src.models.utils import init_weights


class UNetWithInputSkip(nn.Module):
    """
    Disclaimer: we initially experimented with adding a skip connection from the input to the final block, thus the name of this
    class. In that setting, we had an extra block 'final block' after the U-Net decoder that processed the concatenation of the
    input and the decoder output. As it did not bring significant improvements, we removed the skip connection in the final
    version of the code. However, we forgot to remove the final block... Therefore, our final model is equivalent to a standard
    SMP U-Net followed by an extra processing block before the segmentation head. We don't have a good justification for that,
    but we believe it does not harm the performance either.
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
        verbose: bool = True,
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

        # Additional processing after concatenation (SHOULD HAVE BEEN AVOIDED IF NO SKIP...)
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
        self.initialize()

    def _get_activation(self, activation):
        if activation is None or activation == "identity":
            return nn.Identity()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "softmax":
            return nn.Softmax(dim=1)
        else:
            raise ValueError(f"Activation {activation} is not supported")

    def initialize(self):
        # Don't initialize the UNet, just the new layers
        if self.add_skip_connection:
            init_weights(self.skip_input)
        init_weights(self.final_block)
        init_weights(self.segmentation_head)

    def forward(self, x, return_features=False):
        # Get U-Net features
        unet_output = self.unet(x)

        # Skip connection from input
        if self.add_skip_connection:

            x_skip = self.skip_input(x)

            # Concatenate and process
            combined = torch.cat([unet_output, x_skip], dim=1)
            features = self.final_block(combined)
        else:
            features = self.final_block(unet_output)

        if return_features:
            return features

        # Final segmentation
        output = self.segmentation_head(features)

        # Apply activation if specified
        output = self.activation(output)

        return output
