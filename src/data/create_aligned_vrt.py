hi"""
This script corrects xBD raster metadata (CRS and Geotransform) by creating
lightweight .vrt files. These VRTs point to the original .tif files but
contain the correct UTM projections and bounds based on the metadata files.

Warning: If the original .tif files are moved or deleted, the VRTs will break since they reference the original paths.

Usage:
    python src/data/create_aligned_vrt.py --original_xbd_path /path/to/xbd --num_workers 8
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

from osgeo import gdal
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from tqdm import tqdm

from src.data.metadata import load_metadata
from src.utils.time import timeit
from src.utils.geometry import reproject_geo
from src.constants import XBD_S12_PATH


def create_aligned_vrt_file(tif_path: Path, vrt_path: Path, bounds: Tuple[float, float, float, float], crs: str):
    """Creates a VRT file that references a source TIFF but applies new spatial metadata (Bounds and CRS)."""

    xmin, ymin, xmax, ymax = bounds

    with rasterio.open(tif_path) as src:
        # Calculate the affine transform based on fixed bounds and original pixel dimensions
        transform = from_bounds(xmin, ymin, xmax, ymax, src.width, src.height)
        dst_crs = CRS.from_user_input(crs)

    # BuildVRT creates the XML structure pointing to the original TIF
    gdal.BuildVRT(str(vrt_path), [str(tif_path)])

    # Open the VRT in update mode to overwrite the spatial metadata
    ds = gdal.Open(str(vrt_path), gdal.GA_Update)
    if ds is not None:
        ds.SetGeoTransform(transform.to_gdal())
        ds.SetProjection(dst_crs.to_wkt())
        ds = None  # Sync to disk and close


def process_single_uid(row: dict, original_folder: Path, output_folder: Path, overwrite: bool):
    """
    Process both pre- and post-disaster images for a specific UID from the metadata row.
    """
    uid = row["Index"]
    # Calculate corrected spatial info
    true_bounds = reproject_geo(row["geometry"], "EPSG:4326", row["best_utm"]).bounds
    dst_crs = row["best_utm"]

    # Determine tier folder mapping (xBD specific logic)
    tier = "tier1" if row["xbd_tier"] == "train" else row["xbd_tier"]

    tasks = []
    for period in ["pre", "post"]:
        src_path = original_folder / tier / "images" / f"{uid}_{period}_disaster.tif"
        dst_path = output_folder / f"{uid}_{period}_disaster.vrt"  # Saved as .vrt

        if src_path.exists():
            if overwrite or not dst_path.exists():
                create_aligned_vrt_file(src_path, dst_path, true_bounds, dst_crs)
                tasks.append(f"{uid}_{period}")

    return len(tasks)


@timeit
def create_all_aligned_vrt_files(original_folder: Path, overwrite: bool = False, num_workers: int = 8):
    """
    Orchestrates the metadata correction process using multiprocessing.
    """
    output_folder = XBD_S12_PATH / "xbd"
    output_folder.mkdir(exist_ok=True, parents=True)

    df_meta = load_metadata()
    print(f"Processing {len(df_meta)} UIDs using {num_workers} workers...")

    # Convert dataframe rows to a list for the executor
    rows = list(df_meta.itertuples())

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(process_single_uid, row._asdict(), original_folder, output_folder, overwrite) for row in rows]

        # Wrap in tqdm for a progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Creating VRTs"):
            results.append(future.result())

    print(f"All files processed. VRTs saved to: {output_folder}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--original_xbd_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers to use")
    args = parser.parse_args()

    original_xbd_folder = Path(args.original_xbd_path)
    assert original_xbd_folder.exists(), f"Original xBD folder does not exist: {original_xbd_folder}"

    create_all_aligned_vrt_files(original_xbd_folder, overwrite=args.overwrite, num_workers=args.num_workers)
