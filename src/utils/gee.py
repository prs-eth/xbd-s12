"""Utils functions for Google Earth Engine"""

import ee
from shapely.geometry import MultiPolygon, Polygon


def init_gee(project: str, high_volume: bool = False):
    """Initialize GEE. Works also when working through ssh"""

    url = "https://earthengine-highvolume.googleapis.com" if high_volume else None
    try:
        ee.Initialize(url=url, project=project)
    except Exception:
        ee.Authenticate(auth_mode="localhost")
        ee.Initialize(url=url, project=project)


def shapely_to_gee(geo: Polygon | MultiPolygon) -> ee.Geometry:
    """Transforms shapely geometry into GEE geometry"""
    if geo.__geo_interface__["type"] == "Polygon":
        return ee.Geometry.Polygon(geo.__geo_interface__["coordinates"])
    elif geo.__geo_interface__["type"] == "MultiPolygon":
        return ee.Geometry.MultiPolygon(geo.__geo_interface__["coordinates"])
    else:
        raise TypeError


def mosaic_by_date(collection: ee.ImageCollection) -> ee.ImageCollection:
    """Mosaics an ImageCollection by date (assumes date property exists)"""

    # Mosaic only if multiple images
    def get_unique_img(d) -> ee.Image:

        col_date = collection.filter(ee.Filter.eq("date", d))

        def _mosaic_date(col):

            mos = col.mosaic()

            first_img = ee.Image(col.first())
            bands = first_img.bandNames()

            mos = ee.Image(mos.copyProperties(first_img))
            # System properties are not copied by default.
            mos = mos.set("system:index", col.aggregate_array("system:index").join("__"))
            mos = mos.set("system:time_start", first_img.get("system:time_start"))
            mos = mos.set("mosaic", True)

            # Reproject each band to the original projection
            def reproject(bname, mos):
                mos = ee.Image(mos)
                mos_bnames = mos.bandNames()
                bname = ee.String(bname)
                proj = first_img.select(bname).projection()

                newmos = ee.Image(
                    ee.Algorithms.If(
                        mos_bnames.contains(bname),
                        replace_band(mos, bname, mos.select(bname).setDefaultProjection(proj)),
                        mos,
                    )
                )
                return newmos

            return ee.Image(bands.iterate(reproject, mos))

        # Whether the list has one tile or more
        condition = ee.Number(col_date.size()).eq(ee.Number(1))
        return ee.Algorithms.If(condition, col_date.first(), _mosaic_date(col_date))

    unique_dates = collection.aggregate_array("date").distinct()
    return ee.ImageCollection.fromImages(unique_dates.map(get_unique_img))


def replace_band(image, to_replace, to_add):
    """
    From old geetools library

    Replace one band of the image with a provided band

    :param to_replace: name of the band to replace. If the image hasn't got
        that band, it will be added to the image.
    :type to_replace: str
    :param to_add: Image (one band) containing the band to add. If an Image
        with more than one band is provided, it uses the first band.
    :type to_add: ee.Image
    :return: Same Image provided with the band replaced
    :rtype: ee.Image
    """
    # TODO: see Image.addBands({overwrite:True})
    band = to_add.select([0])
    bands = image.bandNames()
    resto = bands.remove(to_replace)
    img_resto = image.select(resto)
    img_final = img_resto.addBands(band)
    return img_final
