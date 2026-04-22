from pathlib import Path

import xarray as xr


def save_raster(xa: xr.DataArray, output_fp: str | Path, if_exists: str = "replace"):
    """Save a raster xarray DataArray to a GeoTIFF file with correct compression."""
    assert if_exists in ["replace", "skip", "raises"], f"if_exists must be one of 'replace', 'skip', or 'raises', got {if_exists}"

    output_fp = Path(output_fp)
    if output_fp.exists() and if_exists == "skip":
        print(f"Output file {output_fp} already exists, skipping.")
        return
    elif output_fp.exists() and if_exists == "replace":
        pass
    elif output_fp.exists() and if_exists == "raises":
        raise FileExistsError(f"Output file {output_fp} already exists.")

    predictor = 2 if xa.dtype.name in ["uint8", "uint16", "int16", "int32", "int64"] else 3  # floating
    output_fp.parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
    xa.rio.to_raster(
        output_fp,
        driver="GTiff",
        compress="DEFLATE",
        predictor=predictor,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="YES",
        NUM_THREADS="ALL_CPUS",
    )
    print(f"Saved raster to {output_fp}.")
