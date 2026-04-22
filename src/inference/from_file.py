import json
import os
from collections.abc import Callable
from functools import reduce
from pathlib import Path

import numpy as np
import rioxarray as rxr
import torch
import torch.utils.data as tdata
from shapely import Geometry
from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import GridGeoSampler
from tqdm import tqdm

from src.constants import S1_BANDS, S2_BANDS, XBD_S12_PATH
from src.inference.base import Inference
from src.inference.utils import prepare_output, save_output_as_input, stitch_prediction_to_output
from src.utils.geometry import reproject_geo


class InferenceFromFiles(Inference):
    """
    Class for running inference from a set of input files (Sentinel-2 and Sentinel-1, pre- and post-event).

    Use ensemble of models if several run names are provided.
    """

    def __init__(
        self,
        run_dmg: str | Path | list[str | Path],
        run_loc: str | Path | list[str | Path],
        patch_size: int = 128,
        padding: int = 32,
        verbose_model: bool = False,
    ):
        """
        Inference from files.

        Args:
            run_dmg (str | Path | list[str | Path]): Name(s) of the damage model run(s) to use for inference.
            run_loc (str | Path | list[str | Path]): Name(s) of the localization model run(s) to use for inference.
            patch_size (int, optional): Size of the patches to use for inference. Defaults to 128.
            padding (int, optional): Padding to use for inference. Defaults to 32.
            verbose_model (bool, optional): Whether to print model details. Defaults to False.
        """
        self.patch_size = patch_size
        self.padding = padding
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__(run_dmg=run_dmg, run_loc=run_loc, verbose_model=verbose_model)

        self.model_dmg_name = [cfg["name"] for cfg in self.cfg_model_dmg][0]

    def run_inference(
        self,
        fp_s2_pre: str | Path,
        fp_s2_post: str | Path,
        fp_s1_pre: str | Path,
        fp_s1_post: str | Path,
        output_fp: str | Path,
        geo: Geometry = None,
    ):
        """
        Run damage assessment inference and write a GeoTIFF.

        All Sentinel images must have been preprocessed at 4m resolution. Optionally, the output can be clipped to
        a provided geometry by passing a shapely geometry in `geo`.

        Args:
            fp_s2_pre (str | Path): Path to Sentinel-2 pre-event image.
            fp_s2_post (str | Path): Path to Sentinel-2 post-event image.
            fp_s1_pre (str | Path): Path to Sentinel-1 pre-event image.
            fp_s1_post (str | Path): Path to Sentinel-1 post-event image.
            output_fp (str | Path): Path to output GeoTIFF that will be written.
            geo (Geometry, optional): If provided, the output is clipped to this geometry. Defaults to None.
        """

        # Check input files
        fp_s2_pre, fp_s2_post = Path(fp_s2_pre), Path(fp_s2_post)
        fp_s1_pre, fp_s1_post = Path(fp_s1_pre), Path(fp_s1_post)
        for fp in [fp_s2_pre, fp_s2_post, fp_s1_pre, fp_s1_post]:
            assert fp.exists(), f"File not found: {fp}"

        # Check output file (to prevent accidental overwriting)
        if output_fp is not None:
            output_fp = Path(output_fp)
            if output_fp.exists():
                print(f"Output file {output_fp} already exists. Please provide a different path or remove the existing file.")
                return

        # Create datasets from files
        ds_s1_pre = SingleRasterDataset(str(fp_s1_pre), all_bands=S1_BANDS, transforms=NormalizeSentinel("s1"))
        ds_s1_post = SingleRasterDataset(str(fp_s1_post), all_bands=S1_BANDS, transforms=NormalizeSentinel("s1"))
        ds_s2_pre = SingleRasterDataset(str(fp_s2_pre), all_bands=S2_BANDS, transforms=NormalizeSentinel("s2"))
        ds_s2_post = SingleRasterDataset(str(fp_s2_post), all_bands=S2_BANDS, transforms=NormalizeSentinel("s2"))
        ds = reduce(lambda x, y: x & y, [ds_s1_pre, ds_s2_pre, ds_s1_post, ds_s2_post])

        stride = self.patch_size - 2 * self.padding
        sampler = GridGeoSampler(ds, size=self.patch_size, stride=stride)
        dataloader = tdata.DataLoader(ds, sampler=sampler, batch_size=4, num_workers=4, collate_fn=stack_samples)
        print(f"Dataloader: {len(dataloader)} batches")

        # Prepare output array
        output, (offset_h, offset_v), transform = prepare_output(fp_s2_pre, bbox=None, verbose=1, n_channels=1, dtype=np.uint8)

        # Inference
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader)):
                bboxes = batch["bounds"]
                img = batch["image"].to(self.device)

                logits_loc = torch.stack([model(img) for model in self.model_locs]).mean(dim=0)

                if self.model_dmg_name == "siamese":
                    img_pre = batch["image"][:, :14].to(self.device)
                    img_post = batch["image"][:, 14:].to(self.device)
                    logits_dmg = torch.stack([model(img_pre, img_post) for model in self.model_dmgs]).mean(dim=0)
                else:
                    logits_dmg = torch.stack([model(img) for model in self.model_dmgs]).mean(dim=0)

                preds_loc = (torch.sigmoid(logits_loc) > 0.5).long().squeeze(1)
                preds_dmg = torch.argmax(logits_dmg[:, 1:], dim=1) + 1
                preds_dmg[preds_loc == 0] = 0
                preds = preds_dmg.cpu().numpy()

                output = stitch_prediction_to_output(
                    preds,
                    bboxes,
                    output,
                    transform,
                    patch_size=self.patch_size,
                    padding=self.padding,
                    offset_h=offset_h,
                    offset_v=offset_v,
                )

        # If no output path is given, return the output array
        if output_fp is None:
            return output

        # Save the output raster, optionally clipped to the provided geometry
        output_fp.parent.mkdir(parents=True, exist_ok=True)
        if geo is not None:
            cutline_wkt = reproject_geo(geo, "EPSG:4326", rxr.open_rasterio(fp_s2_pre).rio.crs).wkt
        else:
            cutline_wkt = None
        save_output_as_input(output, output_fp=output_fp, target_image_path=fp_s2_pre, cutline_wkt=cutline_wkt)
        print(f"Damage map saved → {output_fp}")


class SingleRasterDataset(RasterDataset):
    """A torchgeo dataset that loads a single raster file."""

    def __init__(self, fn: str, transforms: Callable | None = None, all_bands=None, dtype=None, **kwargs):
        """Initialize the SingleRasterDataset class.

        Args:
            fn (str): The path to the raster file.
            transforms (Optional[Callable], optional): The transforms to apply to the
                raster file. Defaults to None.
            all_bands (list, optional): List of all bands in the raster file. Defaults to [].
            dtype (torch.dtype, optional): The dtype to use for the dataset. If None,
                the dtype will be inferred from the data. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        self.original_fp = fn
        self.filename_regex = os.path.basename(fn) + r"$"  # ensure that the filename is matched exactly, and not .tif.aux.xml for instance
        self.all_bands = all_bands if all_bands is not None else []
        self._dtype = dtype
        super().__init__(paths=os.path.dirname(fn), transforms=transforms, **kwargs)

    @property
    def dtype(self) -> torch.dtype:
        """Override the dtype property.

        Returns:
            torch.dtype: The dtype to use for the dataset
        """
        if self._dtype is not None:
            return self._dtype
        # Fall back to parent class behavior if no dtype was specified
        return super().dtype


class NormalizeSentinel:
    def __init__(self, sensor):

        fp_stats = XBD_S12_PATH / "normalization.json"
        assert fp_stats.exists(), f"File {fp_stats} does not exist"
        with open(fp_stats) as f:
            stats = json.load(f)

        self.min_values = torch.tensor(stats[sensor]["1st"])
        self.max_values = torch.tensor(stats[sensor]["99th"])

    def __call__(self, sample):
        img = sample["image"]
        img = (img - self.min_values[:, None, None]) / (self.max_values - self.min_values)[:, None, None]
        img = img.clamp(0, 1)
        sample["image"] = img
        return sample


if __name__ == "__main__":
    from src.constants import DATA_PATH

    folder = DATA_PATH / "inference/palisades_wildfires"
    fp_s2_pre = folder / "s2_2024-11-28_11SLT_l2a_4m.tif"
    fp_s2_post = folder / "s2_2025-02-01_11SLT_l2a_4m.tif"
    fp_s1_pre = folder / "s1_2024-11-27_o137_4m.tif"
    fp_s1_post = folder / "s1_2025-02-07_o137_4m.tif"
    output_fp = folder / "output_from_files_final.tif"

    runs_loc = [f"loc_ce_unet_no_batchnorm_d3_buffer2_s{s}_xview2" for s in [1, 2, 3]]
    runs_dmg = [f"dmg_ce_unet_no_batchnorm_d3_buffer2_s{s}_xview2" for s in [1, 2, 3]]

    infer = InferenceFromFiles(run_dmg=runs_dmg, run_loc=runs_loc, verbose_model=True)
    infer.run_inference(fp_s2_pre, fp_s2_post, fp_s1_pre, fp_s1_post, output_fp)
