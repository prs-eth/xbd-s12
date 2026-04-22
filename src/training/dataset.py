"""The xBD-S12 dataset."""

import json
from pathlib import Path

import geopandas as gpd
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import rioxarray as rxr
import torch
import torch.nn.functional as F
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils import data as tdata

from src.constants import ALL_DISASTERS, S1_BANDS, S2_BANDS, TEST_DISASTERS, TRAIN_DISASTERS, XBD_S12_PATH
from src.training.utils import apply_buffer_around_buildings, downsample_categorical_mask
from src.visualization import plot_mask


class xBDS12Dataset(tdata.Dataset):
    N_LABELS = 7  # background, intact, minor, major, destroyed, unclassified, no-data
    MODALITIES = ["s1", "s2", "s2_tci", "xbd"]
    DEFAULT_DATASET_FOLDER = XBD_S12_PATH  # change if data is stored elsewhere

    def __init__(
        self,
        split: str = "train",
        which_split: str = "event",
        modalities: list | None = None,
        disasters: str | list | None = None,
        transforms: torchvision.transforms.Compose | None = None,
        task: str = "multiclass",  # either 'multiclass' or 'localization'
        fraction_valid: float = 0.15,
        pixels_buffer_around_buildings: int = 3,
        s2_bands: str | list = "all",
        s1_bands: str | list = "all",
        downsample_factor: int | str = 8,
        downsample_xbd: bool = False,
        use_simplified_classes: bool = True,
        use_hdf5: bool = False,
        hdf5_path: str | None = None,
        return_meta: bool = True,
        remove_tiles_without_buildings: bool = False,
        only_pre_disaster: bool = False,
        normalize_data: bool = True,
        verbose: bool = True,
        seed: int = 42,
    ):
        """
        Initialize the xBD Sentinel dataset.

        Args:
            split (str): Which split between train, valid or test. Defaults to "train".
            which_split (str): Which split to use, "xview2" (for original xBD), "event" (from Hafner et al., 2025), or "full" (all disasters).
                Note: which_split=full expects the split to be either 'train' or 'valid', not 'test'. Defaults to "event".
            modalities (list): Which modalities to include, must be a subset of self.MODALITIES. Defaults to ["s1", "s2"].
            disasters (str | list | None): Which disasters to include, must be a subset of ALL_DISASTERS. If None, all disasters are included.
            transforms (torchvision.transforms.Compose | None): Transforms to apply to the data. Defaults to None.
            task (str): Either 'multiclass' for damage classification, or 'localization' for building localization.
                If multiclass, the labels are either 0-2 (if use_simplified_classes is True) or 0-4.
                If localization, the labels are 0 (background) and 1 (building).
                In both case, pixels that are 99 should be ignored.
                Defaults to 'multiclass'.
            fraction_valid (float): Fraction of the training set to use for validation. Defaults to 0.15.
            pixels_buffer_around_buildings (int): Number of pixels to add as buffer around buildings in the mask. Defaults to 3.
            s2_bands (str | list): Which Sentinel-2 bands to use, "all" for all 12 bands, "rgb" for B4,B3,B2,
                or a list of band names (e.g. ["B4", "B3", "B2"]). Defaults to "all".
            s1_bands (str | list): Which Sentinel-1 bands to use, "all" for VV and VH, or a list of band names
                (e.g. ["VV"]). Defaults to "all".
            downsample_factor (int | str): Downsample factor applied to mask (and xBD if downsample_xbd is true)
                If modalities contains s1 or s2, the downsampling factor must be 8. Defaults to 8.
            downsample_xbd (bool): Whether to downsample xBD images by the downsample_factor. Defaults to False.
            use_simplified_classes (bool): Whether to use simplified damage classes (Background, Intact, Damaged) or the original ones.
            use_hdf5 (bool): Whether to read data from a HDF5 file. If True, hdf5_path must be provided. If False, data will be read from
                the default location (XBD_S12_PATH). Defaults to False.
                Note: even when use_hdf5 is True, the metadata and precomputed stats are still expected to be in the DEFAULT_DATASET_FOLDER.
            hdf5_path (str | None): Path to the HDF5 file. Must be provided if use_hdf5 is True. Defaults to None.
            return_meta (bool): Whether to return metadata with the sample. Defaults to True.
            remove_tiles_without_buildings (bool): Whether to remove tiles without buildings. Defaults to False.
            only_pre_disaster (bool): Whether to only include pre-disaster images. This can only be used when task is 'localization'.
                Defaults to False.
            normalize_data (bool): Whether to normalize the data with precomputed stats. Defaults to True.
            verbose (bool): Verbosity. Defaults to True.
            seed (int): Seed for reproducibility. Defaults to 42.
        """

        super().__init__()

        # Check input
        assert split in ["train", "valid", "test"], f"Invalid split {split}"
        assert which_split in ["xview2", "event", "full"], f"Invalid which_split {which_split}"
        if which_split == "full":
            assert split in ["train", "valid"], "which_split=full does not support split=test"
        modalities = ["s1", "s2"] if modalities is None else modalities
        assert all([m in self.MODALITIES for m in modalities]), f"Invalid modalities {modalities}, must be in {self.MODALITIES}"
        if s2_bands == "all":
            s2_bands = S2_BANDS
        elif s2_bands == "rgb":
            s2_bands = ["B4", "B3", "B2"]
        else:
            s2_bands = [s2_bands] if isinstance(s2_bands, str) else s2_bands
        assert all([b in S2_BANDS for b in s2_bands]), f"Invalid s2_bands {s2_bands}, must be in {S2_BANDS}"
        if s1_bands == "all":
            s1_bands = S1_BANDS
        else:
            s1_bands = [s1_bands] if isinstance(s1_bands, str) else s1_bands
        assert all([b in S1_BANDS for b in s1_bands]), f"Invalid s1_bands {s1_bands}, must be in {S1_BANDS}"
        if disasters is not None:
            disasters = [disasters] if isinstance(disasters, str) else disasters
            assert all([d in ALL_DISASTERS for d in disasters]), f"Invalid disasters {disasters}, must be in {ALL_DISASTERS}"
        if any([m in modalities for m in ["s1", "s2", "s2_tci"]]):
            assert downsample_factor == 8, "If using s1 or s2, downsample_factor must be 8"
        if downsample_xbd and downsample_factor is None:
            print("Warning: downsample_xbd is True but downsample_factor is None, nothing will be done")
        assert task in ["multiclass", "localization"], f"Invalid task {task}"
        if use_hdf5:
            assert hdf5_path is not None, "If use_hdf5 is True, hdf5_path must be provided"
            hdf5_path = Path(hdf5_path)
            if not hdf5_path.exists():
                raise FileNotFoundError(f"HDF5 file {hdf5_path} does not exist")
            print(f"Reading data from HDF5 file: {hdf5_path}")
        assert 0 <= fraction_valid <= 1, "fraction_valid must be between 0 and 1"
        assert not (split == "valid" and fraction_valid == 0), "If split is 'valid', fraction_valid must be > 0"
        if only_pre_disaster:
            assert task == "localization", "only_pre_disaster can only be used with localization task"

        # Store parameters
        self.split = split
        self.which_split = which_split
        self.modalities = sorted(modalities)  # sort for consistency
        self.s2_bands = s2_bands
        self.s1_bands = s1_bands
        self.disasters = disasters
        self.downsample_factor = downsample_factor
        self.downsample_xbd = downsample_xbd
        self.transforms = transforms
        self.task = task
        self.use_simplified_classes = use_simplified_classes
        self.use_hdf5 = use_hdf5
        self.hdf5_path = hdf5_path
        self.return_meta = return_meta
        self.remove_tiles_without_buildings = remove_tiles_without_buildings
        self.normalize_data = normalize_data
        self.fraction_valid = fraction_valid
        self.pixels_buffer_around_buildings = pixels_buffer_around_buildings
        self.verbose = verbose
        self.seed = seed

        # Other parameters
        self.s2_bands_ids = [S2_BANDS.index(b) for b in self.s2_bands]
        self.s1_bands_ids = [S1_BANDS.index(b) for b in self.s1_bands]
        self.use_s1 = "s1" in modalities
        self.use_s2 = "s2" in modalities
        self.use_s2_tci = "s2_tci" in modalities
        self.use_xbd = "xbd" in modalities
        self.periods = ["pre", "post"] if not only_pre_disaster else ["pre"]
        if self.return_meta:
            # Columns to return as metadata, for the moment keep it minimal
            self.col_to_return = ["xbd_uid", "disaster"]

        # Check that data and metadata exist
        if not self.use_hdf5:
            instr = "Please run the data preparation steps in the README first."
            # Check that the dataset folder exists (if it does, then we assume metadata and stats are there, as they
            # should come from the Zenodo archive directly)
            if not self.DEFAULT_DATASET_FOLDER.exists():
                raise FileNotFoundError(f"Dataset {self.DEFAULT_DATASET_FOLDER} does not exist. {instr}")
            # Check that the masks have been created
            if not (self.DEFAULT_DATASET_FOLDER / "masks").exists():
                raise FileNotFoundError(f"Masks haven't been created in {self.DEFAULT_DATASET_FOLDER / 'masks'}. {instr}")
            # Check that the xbd images (the corrected ones) have been created
            if self.use_xbd:
                if not (self.DEFAULT_DATASET_FOLDER / "xbd").exists():
                    raise FileNotFoundError(f"Corrected xBD images not found in {self.DEFAULT_DATASET_FOLDER / 'xbd'}. {instr}")
        else:
            # Check that HDF5 file exists
            if not self.hdf5_path.exists():
                raise FileNotFoundError(f"HDF5 file {self.hdf5_path} does not exist. {instr}.")
            # Check that the metadata and stats files exists (should still be in the DEFAULT_DATASET_FOLDER)
            if not (self.DEFAULT_DATASET_FOLDER / "xbd_s12_metadata.geojson").exists():
                raise FileNotFoundError(f"Metadata file not found in {self.DEFAULT_DATASET_FOLDER / 'xbd_s12_metadata.geojson'}. {instr}")
            if not (self.DEFAULT_DATASET_FOLDER / "normalization.json").exists():
                raise FileNotFoundError(f"Normalization file not found in {self.DEFAULT_DATASET_FOLDER / 'normalization.json'}. {instr}")

        # Load metadata
        self.meta = self._load_metadata()
        if self.verbose:
            print(f"Loaded {len(self.meta)} samples for {self.split}")

        # Load precomputed statistics for normalization
        if self.normalize_data:
            self._load_precomputed_stats()

        if self.use_hdf5:
            self.hdf5_file = None  # Will be opened per-worker

    def _get_hdf5_file(self):
        """Open HDF5 file lazily per worker process."""
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, "r")
        return self.hdf5_file

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        sample = self.get_sample_from_row(self.meta.iloc[idx])
        return sample

    def __del__(self):
        """Properly close HDF5 file when worker is destroyed."""
        if self.use_hdf5 and hasattr(self, "hdf5_file") and self.hdf5_file is not None:
            self.hdf5_file.close()
            self.hdf5_file = None

    def get_in_channels(self) -> int:
        """Get the number of input channels based on the selected modalities and bands."""
        n_channels = 0
        if self.use_s2:
            n_channels += len(self.s2_bands)
        if self.use_s2_tci:
            n_channels += 3
        if self.use_s1:
            n_channels += len(self.s1_bands)
        if self.use_xbd:
            n_channels += 3
        return n_channels

    def get_out_channels(self) -> int:
        """Get the number of output channels based on the task."""
        if self.task == "localization":
            return 1  # building vs no-building
        else:
            return 3 if self.use_simplified_classes else 5

    def get_sample_from_row(self, row: pd.Series) -> dict:
        """
        Read data and labels for a given row in the metadata dataframe.

        Args:
            row (pd.Series): Row from the metadata dataframe.

        Returns:
            dict: A dictionary with:
                - images: dict with the images as torch tensors, keys are the modalities
                - labels: torch tensor with the labels (HxW with values in 0-2 or 0-4 (99 for nodata))
                - meta): metadata as a dictionary
        """
        if self.use_hdf5:
            x, y = self.get_sample_from_hdf5(row)
        else:
            x, y = self.get_sample_from_files(row)

        # Normalize data
        if self.normalize_data:
            x = self.normalize(x)

        # Process labels
        if self.task == "localization":
            y[(y > 0) & (y < 5)] = 1  # all buildings to 1
        elif self.use_simplified_classes:
            # merge minor, major, destroyed into 1 class
            y[(y == 3) | (y == 4)] = 2
        if self.downsample_factor is not None and self.downsample_factor > 1:
            y = downsample_categorical_mask(y, self.downsample_factor, self.N_LABELS)
        if self.pixels_buffer_around_buildings > 0:
            y = apply_buffer_around_buildings(y, buffer=self.pixels_buffer_around_buildings)
        y[(y == 5) | (y == 6)] = 99  # unclassified and no-data to 99 (always)

        # Apply transforms
        if self.transforms is not None:
            # Assume it is our custom transform function that takes as inputs a list of images of arbitrary size
            imgs = list(x.values()) + [y]
            imgs_transformed = self.transforms(*imgs)
            x = {k: v for k, v in zip(x.keys(), imgs_transformed[:-1], strict=True)}
            y = imgs_transformed[-1]

        # Store everything in a dict and return
        sample = {"images": x, "labels": y}
        if self.return_meta:
            sample["meta"] = row[self.col_to_return].to_dict()
        return sample

    def get_sample_from_hdf5(self, row: pd.Series) -> tuple:
        """Read data directly from the HDF5 file."""

        hdf5_file = self._get_hdf5_file()
        hdf5_idx = row.hdf5_idx  # get the index to read in hdf5

        # Read images
        x = {}

        for period in self.periods:
            if self.use_s2:
                # Read from HDF5 and convert to float32 tensor
                s2 = torch.from_numpy(hdf5_file[f"s2_{period}"][hdf5_idx]).float()
                if len(self.s2_bands) < 12:
                    s2 = s2[self.s2_bands_ids, :, :]
                x[f"s2_{period}"] = s2

            if self.use_s1:
                s1 = torch.from_numpy(hdf5_file[f"s1_{period}"][hdf5_idx]).float()
                if len(self.s1_bands) < 2:
                    s1 = s1[self.s1_bands_ids, :, :]
                x[f"s1_{period}"] = s1

            if self.use_s2_tci:
                x[f"s2_tci_{period}"] = torch.from_numpy(hdf5_file[f"s2_tci_{period}"][hdf5_idx]).float()

            if self.use_xbd:
                xbd = torch.from_numpy(hdf5_file[f"xbd_{period}"][hdf5_idx]).float()
                if self.downsample_xbd and self.downsample_factor is not None and self.downsample_factor > 1:
                    xbd = F.interpolate(xbd.unsqueeze(0), scale_factor=1 / self.downsample_factor, mode="bilinear").squeeze(0)
                x[f"xbd_{period}"] = xbd

        # Read labels
        y = torch.from_numpy(hdf5_file["mask"][hdf5_idx]).long().squeeze()  # (H, W)

        return x, y

    def get_sample_from_files(self, row: pd.Series) -> tuple:
        """Read data from the original files."""
        uid = row.xbd_uid
        x = {}

        for period in self.periods:
            if self.use_s2:
                fp_s2 = self.DEFAULT_DATASET_FOLDER / "s2" / f"{uid}_{period}_disaster_s2.tif"
                s2 = torch.from_numpy(rxr.open_rasterio(fp_s2).values).float()
                if len(self.s2_bands) < 12:
                    s2 = s2[self.s2_bands_ids, :, :]
                x[f"s2_{period}"] = s2

            if self.use_s1:
                fp_s1 = self.DEFAULT_DATASET_FOLDER / "s1" / f"{uid}_{period}_disaster_s1.tif"
                s1 = torch.from_numpy(rxr.open_rasterio(fp_s1).values).float()
                if len(self.s1_bands) < 2:
                    s1 = s1[self.s1_bands_ids, :, :]
                x[f"s1_{period}"] = s1

            if self.use_s2_tci:
                fp_s2_tci = self.DEFAULT_DATASET_FOLDER / "s2_tci" / f"{uid}_{period}_disaster_s2_tci.tif"
                s2_tci = torch.from_numpy(rxr.open_rasterio(fp_s2_tci).values).float()
                x[f"s2_tci_{period}"] = s2_tci

            if self.use_xbd:
                fp_xbd = self.DEFAULT_DATASET_FOLDER / "xbd" / f"{uid}_{period}_disaster.vrt"
                xbd = torch.from_numpy(rxr.open_rasterio(fp_xbd).values).float()
                if self.downsample_xbd and self.downsample_factor is not None and self.downsample_factor > 1:
                    xbd = F.interpolate(xbd.unsqueeze(0), scale_factor=1 / self.downsample_factor, mode="bilinear").squeeze(0)
                x[f"xbd_{period}"] = xbd

        # Read labels
        fp_mask = self.DEFAULT_DATASET_FOLDER / "masks" / f"{uid}_mask.tif"
        y = torch.from_numpy(rxr.open_rasterio(fp_mask).values).long().squeeze()  # (H, W)

        return x, y

    def _load_metadata(self) -> gpd.GeoDataFrame:
        """Load metadata from the geojson file."""

        # Load metadata
        meta_fp = self.DEFAULT_DATASET_FOLDER / "xbd_s12_metadata.geojson"
        assert meta_fp.exists(), f"Metadata file {meta_fp} does not exist"
        meta = gpd.read_file(meta_fp)

        # keep track of original index to read correct row in hdf5
        if self.use_hdf5:
            # create column hdf5_idx
            meta = meta.reset_index(drop=True).rename_axis("hdf5_idx").reset_index()

        # make sure date are not timestamp (TODO: why ?)
        col_dates = [c for c in meta.columns if "date" in c]
        for col in col_dates:
            # back to string
            meta[col] = meta[col].dt.strftime("%Y-%m-%d")

        # Create the split column based on which_split
        if self.which_split == "xview2":
            # train, tier3 as training, test as testing, discard hold (as in Hafner et al)
            meta.loc[meta.xbd_tier.isin(["train", "tier3"]), "split"] = "train"
            meta.loc[meta.xbd_tier == "test", "split"] = "test"
            meta = meta[meta.xbd_tier != "hold"].copy()
        elif self.which_split == "event":
            meta.loc[meta.disaster.isin(TRAIN_DISASTERS), "split"] = "train"
            meta.loc[meta.disaster.isin(TEST_DISASTERS), "split"] = "test"
        else:
            # full dataset in train split, we can later separate train and valid based on fraction_valid
            meta["split"] = "train"
            pass

        # Keep only the requested split (train/valid/test)
        if self.split in ["train", "valid"]:
            meta = meta[meta.split == "train"]

            if self.fraction_valid > 0:
                # Stratified train/valid split based on disaster
                meta_train, meta_valid = train_test_split(
                    meta,
                    test_size=self.fraction_valid,
                    random_state=self.seed,
                    stratify=meta.disaster,
                )
                meta = meta_valid if self.split == "valid" else meta_train
        elif self.split == "test":
            meta = meta[meta.split == "test"]
        else:
            pass

        # Filter for disasters
        if self.disasters is not None:
            if self.verbose:
                print(f"Filtering for disasters {self.disasters}")
            meta = meta[meta.disaster.isin(self.disasters)]

        if self.remove_tiles_without_buildings:
            if self.verbose:
                print("Removing tiles without buildings")
            meta = meta[meta.N_total > 0]

        return meta.copy()

    def _load_precomputed_stats(self):
        """Load precomputed statistics for normalization (for the correct bands)."""
        fp_stats = self.DEFAULT_DATASET_FOLDER / "normalization.json"
        assert fp_stats.exists(), f"Normalization file {fp_stats} does not exist"
        with open(fp_stats) as f:
            stats = json.load(f)

        self.norm_params = {}
        # Only s1 and s2, the others (s2_tci and xbd) are normalized differently (see normalize function)
        if self.use_s2:
            s2_min = torch.tensor(stats["s2"]["1st"], dtype=torch.float32)
            s2_max = torch.tensor(stats["s2"]["99th"], dtype=torch.float32)
            self.norm_params["s2_min"] = s2_min[self.s2_bands_ids].view(-1, 1, 1)
            self.norm_params["s2_max"] = s2_max[self.s2_bands_ids].view(-1, 1, 1)

        if self.use_s1:
            s1_min = torch.tensor(stats["s1"]["1st"], dtype=torch.float32)
            s1_max = torch.tensor(stats["s1"]["99th"], dtype=torch.float32)
            self.norm_params["s1_min"] = s1_min[self.s1_bands_ids].view(-1, 1, 1)
            self.norm_params["s1_max"] = s1_max[self.s1_bands_ids].view(-1, 1, 1)

    def normalize(self, x: dict) -> dict:
        """
        Normalize the data.

        For RGB data (xBD and Sentinel-2 TCI), we use the normalization (x / 127.5) - 1 to get values between -1 and 1.
        For Sentinel-1 and Sentinel-2, we use min-max normalization based on precomputed 1st and 99th percentiles.
        Data is clamped between 0 and 1 after normalization.

        Args:
            x (dict): dict with the data (eg {"s2_pre": tensor, "s1_post": tensor, ...})

        Returns:
            dict: normalized data
        """
        for k, v in x.items():
            if k.startswith("s2_tci") or k.startswith("xbd"):
                # RGB normalization: (x / 127.5) - 1 to get values between -1 and 1
                x[k] = v.div(127.5).sub(1.0)

            elif k.startswith("s2"):
                min_val = self.norm_params["s2_min"]
                max_val = self.norm_params["s2_max"]
                x[k] = ((v - min_val) / (max_val - min_val + 1e-8)).clamp(0, 1)

            elif k.startswith("s1"):
                min_val = self.norm_params["s1_min"]
                max_val = self.norm_params["s1_max"]
                x[k] = ((v - min_val) / (max_val - min_val + 1e-8)).clamp(0, 1)

        return x

    def unnormalize(self, x: dict) -> dict:
        """
        Unnormalize the data (inverse of self.normalize function). (eg for debugging).

        Args:
            x (dict): dict with the normalized data (eg {"s2_pre": tensor, "s1_post": tensor, ...})

        Returns:
            dict: unnormalized data
        """
        for k, v in x.items():
            if k.startswith("s2_tci") or k.startswith("xbd"):
                # RGB unnormalization: (x + 1) * 127.5
                x[k] = v.add(1.0).mul(127.5)

            elif k.startswith("s2"):
                min_val = self.norm_params["s2_min"]
                max_val = self.norm_params["s2_max"]
                x[k] = v.mul(max_val - min_val).add(min_val)

            elif k.startswith("s1"):
                min_val = self.norm_params["s1_min"]
                max_val = self.norm_params["s1_max"]
                x[k] = v.mul(max_val - min_val).add(min_val)

        return x

    def get_n_imgs(self, add_predictions: bool = False) -> int:
        """Utils for plotting: get the number of images to plot per sample."""
        n_imgs = len(self.modalities) * len(self.periods) + 1  # +1 for the mask
        if add_predictions:
            n_imgs += 1  # +1 for the predictions
        return n_imgs

    def plot(
        self,
        sample: dict,
        axs: list[mpl.axes.Axes] | None = None,
        show=True,
        add_titles: bool = True,
        add_uid_as_ylabel: bool = True,
    ) -> mpl.figure.Figure:
        """
        Plot a sample from the dataset.

        Sample can also contain a new key "predictions" with the model predictions to plot.

        Args:
            sample (dict): Sample from the dataset
            axs (list[mpl.axes.Axes]): Axes to plot on. Defaults to None.
            show (bool): Whether to show the plot. Defaults to True.
            add_titles (bool): Whether to add titles to the plots. Defaults to True.
            add_uid_as_ylabel (bool): Whether to add UID as y-label. Defaults to True.

        Returns:
            mpl.figure.Figure: The figure object containing the plots.
        """

        n_imgs = self.get_n_imgs(add_predictions="predictions" in sample)

        if axs is None:
            fig, axs = plt.subplots(1, n_imgs, figsize=(3 * n_imgs, 3))
        else:
            assert len(axs) == n_imgs, f"Expected {n_imgs} axes, got {len(axs)}"
            fig = None

        # sort modalities (pre, post)
        modalities = sorted(set(["_".join(m.split("_")[:-1]) for m in sample["images"].keys()]))
        imgs = {f"{m}_{p}": sample["images"][f"{m}_{p}"] for m in modalities for p in self.periods}

        # Plot images
        for i, (modality, img) in enumerate(imgs.items()):
            img = img.cpu()

            if modality.startswith("s1"):
                # plot only the first band (VV except if only VH was selected), in gray.
                axs[i].imshow(img[0], cmap="gray")
                if add_titles:
                    axs[i].set_title(f"{modality} ({self.s1_bands[0]})")
            elif modality.startswith("s2_tci") or modality.startswith("xbd"):
                # unnormalize and to RGB
                img_to_plot = img.add(1.0).mul(127.5).permute(1, 2, 0).int()
                axs[i].imshow(img_to_plot)
                if add_titles:
                    axs[i].set_title(f"{modality}")
            elif modality.startswith("s2"):
                # s2 image, B4, B3, B2 as rgb
                if all(b in self.s2_bands for b in ["B4", "B3", "B2"]):
                    b4_id = self.s2_bands.index("B4")
                    b3_id = self.s2_bands.index("B3")
                    b2_id = self.s2_bands.index("B2")
                    img_to_plot = img[[b4_id, b3_id, b2_id], :, :].permute(1, 2, 0)
                    if add_titles:
                        axs[i].set_title(f"{modality} (B4,B3,B2)")
                elif len(self.s2_bands) >= 3:
                    # if 3 bands or more but not RGB, plot the first 3 bands
                    img_to_plot = img[:3, :, :].permute(1, 2, 0)
                    if add_titles:
                        axs[i].set_title(f"{modality} (bands {self.s2_bands[:3]})")
                else:
                    # if only 1 or 2 bands, plot first one
                    img_to_plot = img[0]
                    if add_titles:
                        axs[i].set_title(f"{modality} (band {self.s2_bands[0]})")
                axs[i].imshow(img_to_plot)

        # Plot mask
        plot_mask(sample["labels"].cpu(), ax=axs[i + 1], use_simplified_classes=self.use_simplified_classes)
        if add_titles:
            axs[i + 1].set_title("Labels")

        # Plot predictions if present
        if "predictions" in sample:
            preds = sample["predictions"].cpu()
            plot_mask(preds, ax=axs[i + 2], use_simplified_classes=self.use_simplified_classes)
            if add_titles:
                axs[i + 2].set_title("Predictions")

        # Add UID as y label on the lefftmost plot, and remove axes for the others
        if add_uid_as_ylabel:
            uid = sample["meta"]["xbd_uid"]
            uid = "\n".join(uid.split("_")) + "\n"
            axs[0].set_ylabel(uid, fontsize=14, rotation=90, labelpad=10, verticalalignment="center")
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            [ax.axis("off") for ax in axs[1:]]
        else:
            [ax.axis("off") for ax in axs]

        if show:
            plt.tight_layout()
        return fig


if __name__ == "__main__":
    # Quick test
    dataset = xBDS12Dataset(
        split="test", which_split="event", modalities=["s1", "s2", "s2_tci", "xbd"], task="multiclass", use_simplified_classes=True
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"In channels: {dataset.get_in_channels()}")
    print(f"Out channels: {dataset.get_out_channels()}")

    sample = dataset[0]
    print(sample["images"].keys())
    print(sample["labels"].shape)
    print(sample["meta"])

    dataset.plot(sample)
    plt.show()
