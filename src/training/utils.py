"""Few utils fucntion used across the training codebase."""

import os
import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def downsample_categorical_mask(mask: torch.Tensor, factor: int, n_labels: int) -> torch.Tensor:
    """Downsample a mask by a factor. Must be a perfect divisor of the mask size otherwise we lose information"""
    mask_1h = F.one_hot(mask, num_classes=n_labels).permute(2, 0, 1).float()
    mask_down = F.avg_pool2d(mask_1h.unsqueeze(0), kernel_size=factor, stride=factor)
    return mask_down.squeeze(0).argmax(0).long()  # (W//factor, H//factor)


def apply_buffer_around_buildings(labels: torch.Tensor, buffer: int = 3, nodata_value=99) -> torch.Tensor:
    """
    Apply a buffer around buildings in the labels mask.

    Args:
        labels (torch.Tensor): Tensor of shape (H, W) or (B, H, W) containing building labels.
        buffer (int): Number of pixels to buffer around buildings.
        nodata_value (int): Value to assign to buffered areas.

    Returns:
        torch.Tensor: Tensor with buffered areas set to nodata_value.
    """

    if not buffer:
        return labels

    is_batch = labels.ndim == 3

    if not is_batch:
        labels = labels.unsqueeze(0)  # Add batch dimension

    # Now labels is (B, H, W), need to add channel dim for max_pool2d
    # (B, H, W) -> (B, 1, H, W)
    # building_mask = ((labels > 0) & (labels < 6)).float().unsqueeze(1)
    building_mask = (labels > 0).float().unsqueeze(1)

    # Apply max pooling for dilation
    dilated_mask = F.max_pool2d(building_mask, kernel_size=2 * buffer + 1, stride=1, padding=buffer)

    # Remove channel dimension: (B, 1, H, W) -> (B, H, W)
    dilated_mask = dilated_mask.squeeze(1)
    building_mask = building_mask.squeeze(1)

    # Create buffer mask
    buffer_mask = (dilated_mask > 0) & (building_mask == 0)

    # Apply buffer
    result = torch.where(buffer_mask, nodata_value, labels)

    # Restore original shape if needed
    if not is_batch:
        result = result.squeeze(0)

    return result


def unbind_samples(sample):
    """Adapted from torchgeo.datasets.utils.unbind_samples"""

    def _dict_list_to_list_dict(sample: Mapping[Any, Sequence[Any]]) -> list[dict[Any, Any]]:
        """Convert a dictionary of lists to a list of dictionaries."""
        uncollated: list[dict[Any, Any]] = [{} for _ in range(max(map(len, sample.values())))]
        for key, values in sample.items():
            for i, value in enumerate(values):
                uncollated[i][key] = value
        return uncollated

    for key, values in sample.items():
        if isinstance(values, torch.Tensor):
            sample[key] = torch.unbind(values)
        if isinstance(values, dict):
            sample[key] = _dict_list_to_list_dict(values)
    return _dict_list_to_list_dict(sample)


def seed_everything(seed: int):
    """Set seed for reproducibility. Should be called at the start of the program."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe to call even if cuda not available
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    print(f"Global seed set to {seed}")
