"""xBD-S12 metadata"""

import geopandas as gpd
from src.constants import DATA_PATH


def load_metadata() -> gpd.GeoDataFrame:
    """
    Loads the dataframe containing all the metadata for the xBD-S12 dataset.

    The dataframe contains the following columns:
    TODO: add column descriptions here


    Returns:
        gpd.GeoDataFrame: The metadata
    """

    fp = DATA_PATH / "xbd_s12" / "xbd_s12_metadata.geojson"
    if not fp.exists():
        raise FileNotFoundError(f"Metadata file not found at {fp}. Please run the data preparation steps in the README to download it first.")
    gdf = gpd.read_file(fp).set_index("xbd_uid")
    return gdf


if __name__ == "__main__":

    gdf = load_metadata()
    print(gdf.shape)
    print(gdf.head())
