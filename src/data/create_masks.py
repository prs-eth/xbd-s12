"""
Process xBD JSON label files to generate corresponding masks saved as GeoTIFFs.

The damage classes are encoded as integer values defined in ``DAMAGE_DICT``.
Nodata regions (e.g. black borders from image cropping) are detected from the
aligned VRT files and assigned a dedicated nodata integer value.
"""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rioxarray as rxr
from shapely.wkt import loads as wkt_loads
from tqdm import tqdm

from src.constants import XBD_S12_PATH
from src.data.metadata import load_metadata
from src.utils.time import timeit

# Mapping from original class names to integer labels
DAMAGE_DICT = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 5,  # eg under clouds, or new buildings in post-disaster image
}


@timeit
def create_all_masks(original_folder: Path, overwrite: bool = False, nodata_value: int = 6) -> None:
    """
    Orchestrates the creation of raster masks for all entries in the metadata.

    TODO: Should be parallelized !

    Args:
        original_folder: Path to the root of the original xBD dataset.
        overwrite: If True, regenerates existing masks.
        nodata_value: The integer value representing nodata pixels.
    """
    folder = XBD_S12_PATH / "masks"
    folder.mkdir(exist_ok=True, parents=True)
    df_meta = load_metadata()
    if not overwrite:
        all_exists = all([(folder / f"{uid}_mask.tif").exists() for uid in df_meta.index])
        if all_exists:
            print(f"All masks already exist in {folder}. Skipping mask creation.")
            return

    for row in tqdm(df_meta.itertuples(), total=len(df_meta)):
        uid = row.Index

        out_fp = folder / f"{uid}_mask.tif"
        if out_fp.exists() and not overwrite:
            continue

        tier = "tier1" if row.xbd_tier == "train" else row.xbd_tier
        json_fp = original_folder / tier / "labels" / f"{uid}_post_disaster.json"  # always post
        assert json_fp.exists(), f"JSON file does not exist: {json_fp}"

        create_raster_mask(json_fp, out_fp, nodata_value=nodata_value)

    print(f"All masks created and saved in {folder}")


def create_raster_mask(json_path: Path, out_fp: Path, nodata_value: int = 6) -> None:
    """
    Parses a JSON label file to generate a rasterized mask of building damage.

    Adapted from https://github.com/PaulBorneP/Xview2_Strong_Baseline/blob/master/legacy/create_masks.py


    Args:
        json_path: Path to the post-disaster JSON label file.
        out_fp: Path where the resulting GeoTIFF mask will be saved.
        nodata_value: The integer value to assign to nodata regions.
    """
    # Load the json file and transform to a 1024x1024 mask safely
    with open(json_path) as f:
        data = json.load(f)

    mask = np.zeros((1024, 1024), dtype="uint8")
    for feat in data["features"]["xy"]:
        poly = wkt_loads(feat["wkt"])
        subtype = feat["properties"]["subtype"]
        _mask = mask_for_polygon(poly)

        # Apply damage class to the masked polygon area
        mask[_mask > 0] = DAMAGE_DICT[subtype]

    # Add nodata based on xBD images (eg if the original image is cut)
    uid = out_fp.stem.split("_mask")[0]
    fp_pre = XBD_S12_PATH / "xbd" / f"{uid}_pre_disaster.vrt"
    fp_post = XBD_S12_PATH / "xbd" / f"{uid}_post_disaster.vrt"
    img_pre = rxr.open_rasterio(fp_pre)
    img_post = rxr.open_rasterio(fp_post)

    # Find nodata pixels (are there any valid pixels that are fully black??)
    mask_pre = (img_pre == 0).all(dim="band").values
    mask_post = (img_post == 0).all(dim="band").values
    mask_nodata = (mask_pre + mask_post).astype(int)  # Final mask for nodata values

    # Use one of the images to get the geotransform and crs
    raster = img_post[0]
    raster.rio.set_nodata(0)
    # Apply the nodata value where applicable, otherwise keep the parsed mask
    raster.values = np.where(mask_nodata, nodata_value, mask)

    # Save mask
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    raster.rio.to_raster(out_fp, compress="zstd")


def mask_for_polygon(poly: Any, im_size: tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """
    Converts a Shapely polygon into a 2D binary numpy mask using OpenCV.

    Args:
        poly: A Shapely polygon object.
        im_size: The dimensions of the output mask.

    Returns:
        A 2D numpy array representing the binary mask of the polygon.
    """
    img_mask = np.zeros(im_size, np.uint8)

    def int_coords(x: Any) -> np.ndarray:
        return np.array(x).round().astype(np.int32)

    exteriors = [int_coords(poly.exterior.coords)]
    interiors = [int_coords(pi.coords) for pi in poly.interiors]

    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)

    return img_mask


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--original_xbd_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    original_xbd_folder = Path(args.original_xbd_path)
    assert original_xbd_folder.exists(), f"Original xBD folder does not exist: {original_xbd_folder}"

    create_all_masks(original_xbd_folder, overwrite=args.overwrite)
