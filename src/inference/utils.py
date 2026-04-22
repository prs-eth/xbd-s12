import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.transform import Affine
from shapely.wkt import loads as wkt_loads
from torchgeo.datasets.utils import BoundingBox


def prepare_output(
    input_image_path: str | Path, bbox: BoundingBox = None, verbose: bool = False, n_channels: int = 1, dtype=np.uint8
) -> tuple[np.ndarray, tuple[int], Affine]:
    """
    Prepare output array from an image path.

    If a bounding box is given, the output array will be cropped to the bounding box and the offset adapted accordingly.
    """
    with rasterio.open(input_image_path) as f:
        input_height, input_width = f.shape
        profile = f.profile
        transform = profile["transform"]

    if bbox is not None:
        left, right, top, bottom = bbox_to_img_bounds(bbox, transform)
        input_height = bottom - top
        input_width = right - left
        offset_h = left
        offset_v = top
    else:
        offset_h = 0
        offset_v = 0

    if verbose:
        print(f"Output size: {input_height} x {input_width}")

    output = np.zeros((n_channels, input_height, input_width), dtype=dtype).squeeze()  # if 1 channel only, we return 2D vector

    return output, (offset_h, offset_v), transform


def bbox_to_img_bounds(bbox: BoundingBox, transform: Affine) -> tuple[int]:
    """Transform bounding box into indices in the image space"""
    left, top = ~transform * (bbox.minx, bbox.maxy)
    right, bottom = ~transform * (bbox.maxx, bbox.miny)
    left, right, top, bottom = (
        int(np.round(left)),
        int(np.round(right)),
        int(np.round(top)),
        int(np.round(bottom)),
    )
    return left, right, top, bottom


def stitch_prediction_to_output(
    predictions: np.ndarray,
    bboxes: list[BoundingBox],
    output: np.ndarray,
    transform: Affine,
    patch_size: int,
    padding: int,
    offset_h: int = 0,
    offset_v: int = 0,
):
    """
    Write the predictions for the batch back to the output array.

    The location of the patch in the output array is determined by the bounding box and the transform. The patch is padded on all sides
    except for the borders of the image, where no padding is applied. The padding is removed from the predictions before writing to the output array.

    Args:
        predictions (np.ndarray): The predictions for each patch, of shape (B, C, H, W) or (B, C, W) if n_channels=1.
        bboxes (list[BoundingBox]): The bounding boxes for each patch.
        output (np.ndarray): The output array to write predictions to.
        transform (Affine): The transform to use for converting bounding boxes to image coordinates.
        patch_size (int): The size of each patch.
        padding (int): The amount of padding to apply to each patch.
        offset_h (int, optional): The horizontal offset for the output array. Defaults to 0.
        offset_v (int, optional): The vertical offset for the output array. Defaults to 0.

    Returns:
        _type_: _description_
    """

    if len(output.shape) == 2:
        output = output[np.newaxis, ...]  # add channel dimension if not present

    n_channels, input_height, input_width = output.shape

    for bb, preds in zip(bboxes, predictions, strict=True):
        if len(preds.shape) == 2:
            preds = preds[np.newaxis, ...]  # add channel dimension if not present
        assert preds.shape[0] == n_channels, f"Predictions shape {preds.shape} does not match output shape {output.shape}"

        left, right, top, bottom = bbox_to_img_bounds(bb, transform)
        if offset_h != 0:
            left -= offset_h
            right -= offset_h
        if offset_v != 0:
            top -= offset_v
            bottom -= offset_v

        assert right - left == patch_size, f"Patch size should be equal to the input size but found {right - left} != {patch_size}"
        assert bottom - top == patch_size, f"Patch size should be equal to the input size but found {bottom - top} != {patch_size}"

        # Clamp to image bounds
        left = max(left, 0)
        right = min(right, input_width)
        top = max(top, 0)
        bottom = min(bottom, input_height)

        # Pad everything but borders
        pad_left = 0 if left == 0 else padding
        pad_right = 0 if right == input_width else padding
        pad_top = 0 if top == 0 else padding
        pad_bottom = 0 if bottom == input_height else padding

        # Calculate destination region
        destination_width = (right - pad_right) - (left + pad_left)
        destination_height = (bottom - pad_bottom) - (top + pad_top)

        inp = preds[:, (patch_size - pad_bottom - destination_height) : patch_size - pad_bottom, pad_left : pad_left + destination_width]  # noqa E203
        output[:, top + pad_top : bottom - pad_bottom, left + pad_left : right - pad_right] = inp  # noqa E203

    return output.squeeze()


def save_output_as_input(
    output: np.ndarray,
    output_fp: str | Path,
    target_image_path: str | Path,
    bbox: BoundingBox = None,
    compress: str = "deflate",
    verbose: bool = True,
    cutline_wkt: str = None,
):
    """
    Save the output array as a GeoTIFF file, using the target image as a reference (transform, crs, etc).

    Args:
        output (np.ndarray): The output array to save.
        output_fp (str | Path): The file path to save the output GeoTIFF.
        target_image_path (str | Path): The file path to the target image.
        bbox (BoundingBox, optional): The bounding box to use for cropping. Defaults to None.
        compress (str, optional): The compression method to use. Defaults to "deflate".
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
        cutline_wkt (str, optional): The WKT representation of a cutline to use for masking. Defaults to None.
    """

    # Save predictions
    tic = time.time()

    # Use input image as example
    with rasterio.open(target_image_path) as src:
        profile = src.profile.copy()

        # Bounding box to read only a part of the image
        if bbox is not None:
            window = src.window(bbox.minx, bbox.miny, bbox.maxx, bbox.maxy)
            transform = src.window_transform(window)
            profile["height"] = output.shape[0]
            profile["width"] = output.shape[1]
            profile["transform"] = transform

    profile["driver"] = "GTiff"
    profile["count"] = output.shape[0] if len(output.shape) > 2 else 1  # number of channels
    profile["dtype"] = output.dtype.name
    profile["compress"] = compress
    profile["predictor"] = 2 if output.dtype.name in ["uint8", "uint16", "int16", "int32", "int64"] else 3  # floating
    profile["nodata"] = 0
    profile["tiled"] = True
    profile["blockxsize"] = 512
    profile["blockysize"] = 512
    profile["interleave"] = "pixel"
    profile["bigtiff"] = "YES"

    Path(output_fp).parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
    with rasterio.open(output_fp, "w", **profile) as f:
        if output.ndim == 2:  # single band
            f.write(output, 1)
        elif output.ndim == 3:  # multiple bands
            f.write(output)
        else:
            raise ValueError(f"Unexpected array shape {output.shape}")

    if verbose:
        print(f"Array ({output.shape}) saved in {output_fp} in {time.time() - tic:0.2f} seconds")

    # If cutline is given, we mask everything outside it
    if cutline_wkt is not None:
        print("Masking with cutline...")

        geom = wkt_loads(cutline_wkt)
        with rasterio.open(output_fp) as f:
            out_img, out_transform = mask(f, [geom], crop=True, nodata=0)
            out_meta = f.meta.copy()
            out_meta.update({"height": out_img.shape[1], "width": out_img.shape[2], "transform": out_transform})

        with rasterio.open(output_fp, "w", **out_meta) as dest:
            dest.write(out_img)

        if verbose:
            print(f"Masked and saved again. Array ({output.shape}) saved in {output_fp} in {time.time() - tic:0.2f} seconds")
