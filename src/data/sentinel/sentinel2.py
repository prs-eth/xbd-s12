import datetime as dt
import fnmatch
import io
import warnings
import zipfile
from pathlib import Path

import rioxarray as rxr
import xarray as xr
from phidown import CopernicusDataSearcher
from phidown.downloader import pull_down
from rasterio.enums import Resampling
from shapely import Geometry

from src.constants import PROJECT_PATH
from src.data.sentinel.utils import save_raster

S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]  # warning no B10 in L2A
D_RESOLUTION = {
    "10m": ["B02", "B03", "B04", "B08"],
    "20m": ["B05", "B06", "B07", "B8A", "B11", "B12"],
    "60m": ["B01", "B09", "B10"],
}


def download_sentinel2(
    out_folder: str | Path,
    date: str | dt.date,
    mgrs: str,
    download_safe: bool = True,
    keep_safe_file: bool = False,
    preprocess_tif: bool = True,
    preprocess_tci: bool = False,
    geo: Geometry = None,
    force_recreate: bool = False,
    level: str = "L2A",
    config_file: str | Path = PROJECT_PATH / ".s5cfg",
) -> None:
    """
    Download a Sentinel-2 .SAFE file and optionally preprocess it to GeoTIFF and/or TCI.

    Downloads the .SAFE file for the given MGRS tile and date, then optionally
    preprocesses it to a multi-band GeoTIFF (all spectral bands resampled to 10m)
    and/or a True Color Image (TCI) GeoTIFF. The .SAFE file can optionally be
    deleted after preprocessing.

    Args:
        out_folder (str | Path): The output folder where downloaded and processed files will be saved.
        date (str | dt.date): The acquisition date to download (ISO format string "YYYY-MM-DD" or a datetime.date object).
        mgrs (str): The MGRS tile identifier to download (e.g., "32TNT").
        download_safe (bool): Whether to download the .SAFE file. If False, assumes the .SAFE file is already present in out_folder. Defaults to True.
        keep_safe_file (bool): Whether to keep the .SAFE (or .zip) file after preprocessing. If False, the .SAFE file is deleted once
            preprocessing is complete. Has no effect if no preprocessing is requested. Defaults to False.
        preprocess_tif (bool): Whether to preprocess the .SAFE file into a multi-band GeoTIFF (all spectral bands reprojected to 10m resolution).
            Defaults to True.
        preprocess_tci (bool): Whether to preprocess the .SAFE file into a True Color Image (TCI) GeoTIFF. Defaults to False.
        geo (Geometry): A Shapely Geometry used to clip the output raster(s). If None, the full tile extent is kept. Important, the geometry must be
            in the same CRS as the S2 data. Default to None.
        force_recreate (bool): If True, overwrite existing output files. Defaults to False.
        level (str): Sentinel-2 processing level. Either "L1C" or "L2A". Defaults to "L2A".
        config_file (str | Path): Path to the phidown credentials config file. Defaults to PROJECT_PATH / ".s5cfg".

    Returns:
        None
    """

    outputs = []
    if preprocess_tif:
        tif_out = Path(out_folder) / f"s2_{date}_{mgrs}_{level.lower()}.tif"
        outputs.append(tif_out)
    if preprocess_tci:
        tci_out = Path(out_folder) / f"s2_{date}_{mgrs}_{level.lower()}_TCI.tif"
        outputs.append(tci_out)
    if all([output.exists() for output in outputs]) and not force_recreate:
        print(f"Sentinel-2 file(s) for {mgrs} - {date} already exist, skipping download.")
        return

    out_folder = Path(out_folder)
    date = dt.date.fromisoformat(date) if isinstance(date, str) else date

    # Download .SAFE file
    if download_safe:
        print(f"Downloading Sentinel-2 SAFE file for {mgrs} - {date}...")
        download_sentinel2_safe(
            out_folder=out_folder,
            date=date,
            mgrs=mgrs,
            level=level,
            config_file=config_file,
        )

    # Locate the downloaded .SAFE or .zip file
    safe_files = list(out_folder.glob("*.SAFE")) + list(out_folder.glob("*.zip"))
    if not safe_files:
        print("No .SAFE or .zip file found after download. Skipping preprocessing.")
        return

    # Pick the most recently modified file matching the tile/date
    date_str = date.strftime("%Y%m%d")
    candidates = [f for f in safe_files if mgrs in f.name and date_str in f.name]
    if not candidates:
        print(f"No .SAFE/.zip file found matching mgrs={mgrs}, date={date_str}. Skipping preprocessing.")
        return
    safe_path = candidates[0]

    # Preprocess to multi-band TIF
    if preprocess_tif:
        tif_out = out_folder / f"s2_{date}_{mgrs}_{level.lower()}.tif"
        print("Preprocessing to multi-band raster at 10m resolution (.tif)...")
        process_s2_safe(
            input_dir=safe_path,
            output_dir=tif_out,
            target="tif",
            geometry=geo,
            force_recreate=force_recreate,
        )

    # Preprocess to TCI
    if preprocess_tci:
        tci_out = out_folder / f"s2_{date}_{mgrs}_{level.lower()}_TCI.tif"
        print("Preprocessing to True Color Image (.tif)...")
        process_s2_safe(
            input_dir=safe_path,
            output_dir=tci_out,
            target="tci",
            geometry=geo,
            force_recreate=force_recreate,
        )

    # Remove .SAFE file if requested
    if not keep_safe_file and (preprocess_tif or preprocess_tci):
        print(f"Removing .SAFE file: {safe_path}")
        if safe_path.is_dir():
            import shutil

            shutil.rmtree(safe_path)
        else:
            safe_path.unlink()


def download_sentinel2_safe(
    out_folder: str | Path,
    date: str | dt.date,
    mgrs: str,
    level: str = "L2A",
    searcher: CopernicusDataSearcher = None,
    config_file: str | Path = PROJECT_PATH / ".s5cfg",
):
    """
    Download Sentinel-2 .SAFE file for the given date and MGRS tile using phidown.

    Args:
        out_folder (str | Path): The output folder.
        date (str | dt.date): The date to download (YYYY-MM-DD).
        mgrs (str): The MGRS tile to download (e.g., "14QMG").
        level (str): The processing level to download. Either "L1C" or "L2A". Defaults to "L2A".
        searcher (CopernicusDataSearcher): An existing searcher to use. If None, a new one will be created.
            Defaults to None.
        config_file (str | Path): The configuration file to use. Defaults to PROJECT_PATH/".s5cfg".
    """
    assert level in ["L1C", "L2A"], "level must be either 'L1C' or 'L2A'"

    assert Path(config_file).exists(), f"Config file with credentials {config_file} does not exist. Please see phidown's README"

    date = dt.date.fromisoformat(date) if isinstance(date, str) else date
    if searcher is None:
        searcher = CopernicusDataSearcher()

    # Search
    searcher.query_by_filter(
        collection_name="SENTINEL-2",
        product_type="S2MSI2A" if level == "L2A" else "S2MSI1C",
        attributes={"tileId": mgrs},
        start_date=date.isoformat(),
        end_date=(date + dt.timedelta(days=1)).isoformat(),
    )
    df = searcher.execute_query()

    # Check results
    if df.empty:
        print(f"No data found for {mgrs} - {date}")
        return
    if len(df) > 1:
        # Happens rarely (I beleive when multiple processing have been archived...)
        print(f"Found {len(df)} results (mgrs={mgrs}, date={date})")
        print("Taking largest file...")
        df = df.loc[[df["ContentLength"].idxmax()]]

    # Download
    pull_down(s3_path=df.S3Path.iloc[0], output_dir=out_folder, config_file=config_file)
    print(f"Downloaded .SAFE file for {mgrs} - {date}")


def process_s2_safe(input_dir: str | Path, output_dir: str | Path, target: str = "tif", geometry: Geometry = None, force_recreate: bool = False):
    """
    Process a Sentinel-2 .SAFE file (zipped or unzipped).

    Transform it to either the full S2 tile (12 bands, 10m reprojected), or just extract the
    True Color Image (TCI) band. If geometry is given, clip the output to the geometry.

    Args:
        input_dir (str | Path): Path to the .SAFE file (zipped or unzipped).
        output_dir (str | Path): Path to the output .tif file.
        target (str): Either tif or tci. Defaults to "tif".
        geometry (Geometry): The geometry. Defaults to None.
        force_recreate (bool): Force recreation when file already exists. Defaults to False.
    """

    # Check paths
    input_dir = Path(input_dir)  # Either .zip or .SAFE
    assert input_dir.exists(), f"Input {input_dir} does not exist"
    if output_dir.exists() and not force_recreate:
        print(f"{output_dir} already exists")
        return
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir)
    if target == "tif":
        s2 = process_s2_safe_to_tif(input_dir)
    elif target == "tci":
        s2 = process_s2_safe_to_tci(input_dir)
    else:
        raise ValueError(f"target must be one of ['tif', 'tci'], got {target}")

    # Clip to geometry if given
    if geometry:
        if not hasattr(geometry, "__iter__"):
            geometry = [geometry]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s2 = s2.rio.clip(geometry, all_touched=True, drop=True)

    # Save
    save_raster(s2, output_dir)
    print(f"Wrote {output_dir}")


def process_s2_safe_to_tif(input_dir: str | Path, resampling: Resampling = Resampling.bilinear) -> xr.DataArray:
    """
    Read all bands from a Sentinel-2 .SAFE file (zipped or unzipped), reproject them to 10m resolution.

    Args:
        input_dir (str | Path): Path to the .SAFE file
        resampling (Resampling): Resampling method. Defaults to Resampling.biline

    Returns:
        xr.DataArray: The 12 bands of the Sentinel-2 tile.
    """

    # Read each band
    d_bands = {}

    level = input_dir.name.split("_")[1][3:]  # L1C or L2A
    if level == "L2A":
        bands_to_read = [b for b in S2_BANDS if b != "B10"]  # no B10 in L2A
    else:
        bands_to_read = S2_BANDS

    for name in bands_to_read:
        if level == "L2A":
            resolution = "_10m" if name in D_RESOLUTION["10m"] else "_20m" if name in D_RESOLUTION["20m"] else "_60m"
        else:
            resolution = ""  # no resolution suffix in L1C

        # Read band from file (either zip or .SAFE)
        if str(input_dir).endswith(".zip"):
            fps = glob_search_in_zip(input_dir, f"**/IMG_DATA/**/*{name}{resolution}.jp2")
            d_bands[name] = read_tif_from_zip(input_dir, fps[0])
        else:
            fps = list(input_dir.glob(f"**/IMG_DATA/**/*{name}{resolution}.jp2"))
            d_bands[name] = rxr.open_rasterio(fps[0], chunks=True)

    # Resample all bands to the same resolution (10m)
    ref_band = d_bands["B02"]
    d_resampled_bands = {}
    for name, band in d_bands.items():
        if band.shape == ref_band.shape:
            d_resampled_bands[name] = band
        else:
            d_resampled_bands[name] = band.rio.reproject_match(ref_band, resampling=resampling)

    # Concatenate all bands
    bands_variable = xr.Variable("band", list(d_resampled_bands.keys()))
    s2 = xr.concat([band.squeeze() for band in d_resampled_bands.values()], dim=bands_variable)

    del d_bands
    del d_resampled_bands
    return s2


def process_s2_safe_to_tci(input_dir: str | Path) -> xr.DataArray:
    """
    Read the True Color Image (TCI) band from a Sentinel-2 .SAFE file (zipped or unzipped).

    Args:
        input_dir (str | Path): Path to the .SAFE file

    Returns:
        xr.DataArray: The True Color Image band.
    """

    level = input_dir.name.split("_")[1][3:]  # L1C or L2A
    resolution = "_10m" if level == "L2A" else ""

    # Read TCI from file (either zip or .SAFE)
    if str(input_dir).endswith(".zip"):
        fp = glob_search_in_zip(input_dir, f"**/*TCI{resolution}.jp2")[0]
        tci = read_tif_from_zip(input_dir, fp)
    else:
        fp = list(input_dir.glob(f"**/*TCI{resolution}.jp2"))[0]
        tci = rxr.open_rasterio(fp, chunks=True)

    return tci


def glob_search_in_zip(zip_filename: str | Path, pattern: str) -> list[str]:
    """.glob method for zip files."""
    matches = []
    with zipfile.ZipFile(zip_filename, "r") as zipf:
        for file_name in zipf.namelist():
            if fnmatch.fnmatch(file_name, pattern):
                matches.append(file_name)
    return matches


def read_tif_from_zip(zip_filename: str | Path, tif_filename: str) -> xr.DataArray:
    """Read a .tif file from a .zip file."""
    with zipfile.ZipFile(zip_filename, "r") as zipf:
        with zipf.open(tif_filename) as tif_file:
            tif_content = io.BytesIO(tif_file.read())
            tif_data = rxr.open_rasterio(tif_content, chunks=True)
            return tif_data
