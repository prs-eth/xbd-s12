from pyproj import Transformer
from shapely.ops import transform
from shapely.geometry.base import BaseGeometry


def reproject_geo(geo: BaseGeometry, current_crs: str, target_crs: str) -> BaseGeometry:
    """Reprojects a Shapely geometry from the current CRS to a new CRS."""
    transformer = Transformer.from_crs(current_crs, target_crs, always_xy=True)
    return transform(transformer.transform, geo)
