import torch.nn as nn


def init_weights(module: nn.Module) -> None:
    """
    Recursively initializes weights for all sub-modules in a given nn.Module.

    - Conv2d: Kaiming Uniform (He initialization) with ReLU nonlinearity.
    - Normalization layers: Weights set to 1, biases to 0.
    - Linear: Xavier Uniform (Glorot initialization).
    - Biases: All set to 0 where applicable.

    Args:
        module: The PyTorch module to initialize.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
