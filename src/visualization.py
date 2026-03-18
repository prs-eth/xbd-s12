from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr
import xarray as xr

from src.constants import CLASSES, CLASSES_ORIGINAL, COLORS, COLORS_ORIGINAL


def create_damage_colormap(use_simplified_classes=False):
    """Create a custom colormap for damage visualization."""

    d_colors = COLORS if use_simplified_classes else COLORS_ORIGINAL
    values = sorted(d_colors.keys())
    colors = [d_colors[v] for v in values]

    # Create colormap
    cmap = mpl.colors.ListedColormap(colors)

    # Create normalization that maps values to colormap indices
    boundaries = values + [max(values) + 1]  # Add boundary for last color
    boundaries = [b - 0.5 for b in boundaries]  # Center boundaries between values
    norm = mpl.colors.BoundaryNorm(boundaries, cmap.N)

    return cmap, norm, values


def plot_mask(mask: Any, ax: mpl.axes.Axes = None, add_colorbar: bool = False, use_simplified_classes: bool = False) -> mpl.axes.Axes:
    """
    Plot damage mask with custom colormap.
    Args:
        mask (Any): The mask array. Can be a file path, rioxarray.DataArray, numpy array or torch tensor. If 3D, assumed to be one-hot encoded.
        ax (mpl.axes.Axes, optional): The matplotlib axes to plot on. Defaults to None.
        add_colorbar (bool, optional): Whether to add a colorbar to the plot. Defaults to False.
        alpha (float, optional): The alpha transparency level for the plot. Defaults to 1.
        use_simplified_classes (bool, optional): Whether we use our simplified damage classes (Background, Intact, Damaged) or the original ones.

    Returns:
        mpl.axes.Axes: The matplotlib axes with the plotted mask.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    # Transform into numpy array
    if isinstance(mask, str):
        mask = Path(mask)
    if isinstance(mask, Path):
        assert mask.exists()
        mask = rxr.open_rasterio(mask)
    if isinstance(mask, xr.DataArray):
        mask = mask.values
    img = mask.squeeze()

    # Check if 3D
    if len(img.shape) == 3:
        # Assume mask is one-hot encoded
        img = np.argmax(img, axis=0)
    else:
        pass

    # Create colormap and normalization
    cmap, norm, values = create_damage_colormap(use_simplified_classes=use_simplified_classes)
    im = ax.imshow(img, cmap=cmap, norm=norm, interpolation="nearest")

    if add_colorbar:
        classes = CLASSES if use_simplified_classes else CLASSES_ORIGINAL
        fig = plt.gcf()
        cbar = fig.colorbar(
            im,
            ticks=[i + 0.5 for i in range(len(classes))],
            fraction=0.046,
            pad=0.04,
        )
        cbar.ax.set_yticklabels(list(values))
    return ax
