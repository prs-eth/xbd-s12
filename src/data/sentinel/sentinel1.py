import datetime as dt
from pathlib import Path

import ee
import geemap
from retry import retry
from shapely import Geometry

from src.utils.gee import mosaic_by_date, shapely_to_gee


@retry(tries=10, delay=1, backoff=2)
def download_sentinel1(
    folder: str | Path,
    date: str | dt.date,
    orbit_number: int,
    geo: Geometry | ee.Geometry,
    bands: list[str] = None,
    crs: str = None,
    clip_to_geo: bool = True,
    force_recreate: bool = False,
):
    """
    Download Sentinel-1 data using Google Earth Engine.

    This function downloads Sentinel-1 data for a specific ID, or a specific date, orbit number, and geometry.
    If the geometry is given, it will clip the image to that geometry.

    Args:
        folder (str | Path): Path to save the downloaded image.
        date (str | dt.date): Date of the image. Can be a string in the format "YYYY-MM-DD" or a datetime.date object.
        orbit_number (int): Orbit number of the image.
        geo (Geometry | ee.Geometry): Geometry to filter the image collection and optionally clip the image.
            Can be a shapely Geometry or an ee.Geometry.
        bands (list[str], optional): List of bands to keep. Defaults to None (keep all).
        crs (str, optional): Coordinate reference system to reproject the image. Defaults to None (use original one).
        clip_to_geo (bool, optional): Whether to clip the image to the given geometry. If true, geo must be provided.
            Defaults to True.
        force_recreate (bool, optional): Whether to force re-download even if the file exists. Defaults to False.
    """

    folder = Path(folder) if isinstance(folder, str) else folder
    filename = folder / f"s1_{date}_o{orbit_number}.tif"

    date = date.isoformat() if isinstance(date, dt.date) else date
    if filename.exists() and not force_recreate:
        print(f"File {filename.name} already exists, skipping download.")
        return
    if isinstance(geo, Geometry):
        geo = shapely_to_gee(geo)

    s1 = get_s1_collection(date, ee.Date(date).advance(1, "day"), geo=geo, orbit_number=orbit_number)

    # Check number of images found
    n_imgs = s1.size().getInfo()
    if n_imgs == 0:
        print(f"No images found for {date=}, {orbit_number=}")
        return
    elif n_imgs > 1:
        # Can happen if id_ is None or multiple IDs, (TODO: check if desired behavior)
        print(f"Multiple images found for {date=}, {orbit_number=}. Mosaicking them together...")
        img = mosaic_by_date(s1).first()
    else:
        img = s1.first()

    # Keep only some bands
    if bands is not None:
        print(f"Keeping only bands {bands}...")
        img = img.select(bands)

    # Download
    filename.parents[0].mkdir(parents=True, exist_ok=True)
    clip_geo = geo if clip_to_geo else None
    crs = img.select(0).projection().crs().getInfo() if crs is None else crs
    geemap.download_ee_image(img, filename, region=clip_geo, scale=10, dtype="float32", crs=crs)


def get_s1_collection(
    start_date: str | dt.date | ee.Date = "2014-10-03",
    end_date: str | dt.date | ee.Date = None,
    geo: Geometry | ee.Geometry = None,
    direction: str = None,
    orbit_number: int = None,
    id_: str | list[str] = None,
):
    """
    Get a Sentinel-1 image collection from Google Earth Engine.

    Args:
        start_date (str | dt.date | ee.Date, optional): The start date for the image collection. Defaults to "2014-10-03".
        end_date (str | dt.date | ee.Date, optional): The end date for the image collection. Defaults to dt.date.today().
        geo (Geometry | ee.Geometry, optional): The geographic area to filter the image collection. Defaults to None.
        direction (str, optional): The orbit direction to filter the image collection. Defaults to None.
        orbit_number (int, optional): The orbit number to filter the image collection. Defaults to None.
        id_ (str | list[str], optional): The image ID(s) to filter the image collection. If provided, bypass all other
            filters. Defaults to None.

    Returns:
        ee.ImageCollection: The filtered Sentinel-1 image collection.
    """
    if end_date is None:
        end_date = dt.date.today()

    # Check inputs
    geo = shapely_to_gee(geo) if isinstance(geo, Geometry) else geo
    start_date = start_date.isoformat() if isinstance(start_date, dt.date) else ee.Date(start_date)
    end_date = end_date.isoformat() if isinstance(end_date, dt.date) else ee.Date(end_date)

    # Query the collection
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    if id_ is not None:
        if isinstance(id_, str):
            # a single id_ precisely identifies a single image
            s1 = s1.filter(ee.Filter.eq("system:index", id_))
        else:
            s1 = s1.filter(ee.Filter.inList("system:index", id_))
    else:
        s1 = s1.filterDate(start_date, end_date)
        if geo is not None:
            s1 = s1.filterBounds(geo)
        if orbit_number is not None:
            s1 = s1.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit_number))
        if direction is not None:
            s1 = s1.filter(ee.Filter.eq("orbitProperties_pass", direction))

    # Add some metadata
    s1 = s1.map(
        lambda img: img.set(
            {
                "id_": img.id(),
                "date": img.date().format("YYYY-MM-dd"),
                "orbit_number": img.get("relativeOrbitNumber_start"),
                "orbit_direction": img.get("orbitProperties_pass"),
            }
        )
    )
    return s1.sort("system:time_start")
