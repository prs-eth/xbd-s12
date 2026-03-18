# xBD-S12 Dataset Description

The dataset is organized around the `xbd_s12_metadata.geojson` file, which contains all relevant metadata for each patch. Below is a description of the key columns:

| Column | Description |
|---|---|
| `geometry` | Polygon geometry of the patch in geographic coordinates |
| `patch_id` | Unique identifier for each patch |
| `disaster` | Name of the disaster event associated with the patch |
| `disaster_type` | Category of the disaster (e.g., flood, wildfire, earthquake) |
| `split` | Dataset split assignment (`train`, `val`, or `test`) |
| `damage_class` | Dominant damage level label for the patch |
| `s1_file` | Relative path to the corresponding Sentinel-1 file |
| `s2_file` | Relative path to the corresponding Sentinel-2 file |
| `tci_file` | Relative path to the corresponding Sentinel-2 TCI (RGB) file |
| `mask_file` | Relative path to the raster damage mask file |
| `xbd_tile` | Name of the original xBD tile from which the patch was derived |
| `pre_post` | Indicates whether the imagery is pre- or post-disaster |
| `date` | Acquisition date of the satellite imagery |
| `cloud_cover` | Estimated cloud cover percentage for the patch |
| `crs` | Coordinate reference system of the patch |
| `gsd` | Ground sampling distance in meters (~4 m after resampling) |

> **Note:** Column descriptions are inferred from naming conventions — please update this table to reflect the actual schema.

### Modalities

Three image modalities are included, each patch resampled (Lanczos) and cropped to **128×128 pixels** at ~4 m GSD, matching the xBD tile extent:

- **Sentinel-1** — Downloaded directly from Google Earth Engine. Amplitude bands (VV, VH) expressed in decibels. Occasional mosaicking applied where necessary.
- **Sentinel-2** — Downloaded using the [phi-down](https://github.com/ESA-PhiLab/phidown) library. All bands resampled to 10 m resolution. Values are in original reflectance scale.
- **Sentinel-2 TCI** — RGB composite used for visualization only. Also sourced via phi-down. No further processing; values are `uint8` [0–255].

The original (pre-tiling) satellite tiles are available on Zenodo at [zenodo_url].

---

## Quick Exploration

The notebook [`xbd_s12_exploration.ipynb`](xbd_s12_exploration.ipynb) provides a quick overview of the dataset, including patch visualization and metadata statistics.