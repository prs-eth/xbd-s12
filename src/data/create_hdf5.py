"""Script to generate the HDF5 file."""

from multiprocessing import Pool, cpu_count
from pathlib import Path

import h5py
import numpy as np
import rasterio
from tqdm import tqdm

from src.constants import XBD_S12_PATH
from src.data.metadata import load_metadata
from src.utils.time import timeit

# Modalities to process (Note: xBD is not needed to train a model, so we don't include it by default)
MODALITIES = ["s2", "s1", "s2_tci"]  # xbd
CHUNK_SIZE = 32  # Number of samples to read in parallel per batch. Shjould be adjusted


@timeit
def create_hdf5_file(output_path: str | Path, num_workers: int | None = None, overwrite: bool = False) -> None:
    """
    Creates an HDF5 file containing all images and masks for the xBD-S12 dataset.

    Arrays are stored compressed and chunked by sample (chunk size is (1, C, H, W)
    for images and (1, H, W) for masks) to allow for efficient reading.

    Reads are parallelised across workers; writes remain serial since h5py
    is not thread/process safe.

    Args:
        output_path: Path where the HDF5 file will be saved.
        num_workers: Number of worker processes. Defaults to cpu_count() - 1.
        overwrite: If True, will overwrite existing HDF5 file at output_path.
    """
    if not overwrite and Path(output_path).exists():
        print(f"HDF5 file already exists at {output_path}. Skipping HDF5 creation.")
        return

    num_workers = num_workers or max(1, cpu_count() - 1)
    print(f"Creating HDF5 file with {num_workers} workers...")

    meta = load_metadata()
    meta = meta.reset_index().rename_axis("hdf5_idx").reset_index()
    n_samples = len(meta)
    print(f"Metadata loaded with {n_samples} samples.")

    # Infer shapes and dtypes from the first sample
    first_uid = meta.iloc[0].xbd_uid
    shapes, dtypes = {}, {}
    for mod in MODALITIES:
        fp = get_fp(first_uid, mod, "pre")
        shapes[f"{mod}_pre"], dtypes[f"{mod}_pre"] = get_shape_dtype(fp)
        shapes[f"{mod}_post"] = shapes[f"{mod}_pre"]
        dtypes[f"{mod}_post"] = dtypes[f"{mod}_pre"]
    shapes["mask"], dtypes["mask"] = get_shape_dtype(get_fp(first_uid, "mask"))

    for key in shapes:
        print(f"  {key}: shape={shapes[key]}, dtype={dtypes[key]}")

    # Build the list of (idx, uid) work items
    work_items: list[tuple[int, str]] = [(row.hdf5_idx, row.xbd_uid) for _, row in meta.iterrows()]

    with h5py.File(output_path, "w") as f:
        # Pre-allocate all datasets
        for key, shape in shapes.items():
            f.create_dataset(
                key,
                shape=(n_samples, *shape),
                dtype=dtypes[key],
                chunks=(1, *shape),
                compression="lzf",
            )

        # Read in parallel, write serially in chunk batches
        with Pool(processes=num_workers) as pool:
            for batch_start in tqdm(range(0, n_samples, CHUNK_SIZE), desc="Batches"):
                batch = work_items[batch_start : batch_start + CHUNK_SIZE]  # noqa: E203
                results: list[tuple[int, dict]] = pool.map(read_sample, batch)

                # Write results in index order (cosmetic — HDF5 allows random writes)
                for idx, data in sorted(results, key=lambda x: x[0]):
                    for key, arr in data.items():
                        f[key][idx] = arr

    print(f"HDF5 file created at: {output_path}")


def get_fp(uid: str, mod: str, period: str | None = None) -> Path:
    if mod == "mask":
        fp = XBD_S12_PATH / "masks" / f"{uid}_mask.tif"
    elif mod == "xbd":
        fp = XBD_S12_PATH / "xbd" / f"{uid}_{period}_disaster.vrt"
    else:
        fp = XBD_S12_PATH / mod / f"{uid}_{period}_disaster_{mod}.tif"
    return fp


def read_tif(fp: str | Path) -> np.ndarray:
    with rasterio.open(fp) as src:
        return src.read().squeeze()


def get_shape_dtype(fp: str | Path) -> tuple[tuple[int, ...], np.dtype]:
    arr = read_tif(fp)
    return arr.shape, arr.dtype


def read_sample(sample_row: tuple) -> tuple[int, dict[str, np.ndarray]]:
    """
    Read all arrays for a single sample. Designed to be called in a worker process.

    Args:
        sample_row: Tuple of (hdf5_idx, xbd_uid) for the sample.

    Returns:
        Tuple of (hdf5_idx, dict mapping dataset key -> numpy array).
    """
    idx, uid = sample_row
    data = {}
    for mod in MODALITIES:
        data[f"{mod}_pre"] = read_tif(get_fp(uid, mod, "pre"))
        data[f"{mod}_post"] = read_tif(get_fp(uid, mod, "post"))
    data["mask"] = read_tif(get_fp(uid, "mask"))
    return idx, data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create HDF5 file for xBD-S12 dataset.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes to use. Defaults to cpu_count() - 1.")
    args = parser.parse_args()

    output_path = XBD_S12_PATH / "xbd_s12.hdf5"
    create_hdf5_file(output_path, num_workers=args.num_workers)
