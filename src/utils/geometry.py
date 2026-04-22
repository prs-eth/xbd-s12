"""Geometry utilities."""

import geopandas as gpd
from pyproj import Transformer
from shapely import Geometry, Point
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform


def reproject_geo(geo: BaseGeometry, current_crs: str, target_crs: str) -> BaseGeometry:
    """Reprojects a Shapely geometry from the current CRS to a new CRS."""
    transformer = Transformer.from_crs(current_crs, target_crs, always_xy=True)
    return transform(transformer.transform, geo)


def get_best_utm_crs(shape: Geometry | gpd.GeoDataFrame) -> str:
    """Get the best UTM CRS for the given shape."""
    if isinstance(shape, Geometry):
        if not isinstance(shape, Point):
            shape = shape.centroid
        lon, lat = shape.x, shape.y
    elif isinstance(shape, gpd.GeoDataFrame):
        lon = box(*shape.total_bounds).centroid.x
        lat = box(*shape.total_bounds).centroid.y
    else:
        raise ValueError(f"Invalid shape type: {type(shape)}")
    return get_best_utm_from_lon_lat(lon, lat)


def get_best_utm_from_lon_lat(lon: float, lat: float) -> str:
    utm_zone = int(((lon + 180) / 6) % 60) + 1
    utm_zone_str = str(utm_zone).zfill(2)
    utm_crs = f"EPSG:326{utm_zone_str}" if lat > 0 else f"EPSG:327{utm_zone_str}"
    return utm_crs
