from pyproj import Transformer
from shapely.ops import transform


def reproject_geo(geo, current_crs, target_crs):
    """Reprojects a Shapely geometrz from the current CRS to a new CRS."""
    transformer = Transformer.from_crs(current_crs, target_crs, always_xy=True)
    return transform(transformer.transform, geo)
