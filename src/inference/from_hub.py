from functools import reduce
from pathlib import Path

import numpy as np
import rioxarray as rxr
import torch
import torch.nn as nn
import torch.utils.data as tdata
from shapely import Geometry
from torchgeo.datasets import stack_samples
from torchgeo.samplers import GridGeoSampler
from tqdm import tqdm

from src.constants import S1_BANDS, S2_BANDS
from src.inference.from_file import SingleRasterDataset
from src.inference.utils import prepare_output, save_output_as_input, stitch_prediction_to_output
from src.models.siamese import SiameseUnetLateFusion
from src.models.unet import UNetWithInputSkip
from src.utils.geometry import reproject_geo

# Names of models inside the HuggingFace repo (see https://huggingface.co/collections/prs-eth/xbd-s12)
DEFAULT_LOC_MODELS = ["prs-eth/xbd-s12_loc_seed1", "prs-eth/xbd-s12_loc_seed2", "prs-eth/xbd-s12_loc_seed3"]
DEFAULT_DMG_MODELS = ["prs-eth/xbd-s12_dmg_seed1", "prs-eth/xbd-s12_dmg_seed2", "prs-eth/xbd-s12_dmg_seed3"]


class InferenceFromHub:
    """End-to-end inference pipeline from HuggingFace."""

    def __init__(
        self,
        loc_models: list[str] = None,
        dmg_models: list[str] = None,
        patch_size: int = 128,
        padding: int = 32,
    ):
        """
        Inference from HuggingFace Hub.

        Args:
            loc_models (list[str]): List of localization model names.
                Defaults to the three seeds shipped in HF (prs-eth/xbd-s12_loc_seedX).
            dmg_models (list[str]): List of damage model names.
                Defaults to the three seeds shipped in HF (prs-eth/xbd-s12_dmg_seedX).
            patch_size (int): Spatial size of inference tiles (pixels). Defaults to 128.
            padding (int): Overlap padding to remove tile-boundary artifacts. Defaults to 32.
        """
        self.patch_size = patch_size
        self.padding = padding
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        loc_models = loc_models or DEFAULT_LOC_MODELS
        dmg_models = dmg_models or DEFAULT_DMG_MODELS

        # ── Load localization models ───────────────────────────────────────────
        print(f"Loading {len(loc_models)} localization model(s) ...")
        self.model_locs = [self._load_model(model_name, "loc").to(self.device).eval() for model_name in loc_models]

        # ── Load damage models ─────────────────────────────────────────────────
        print(f"Loading {len(dmg_models)} damage model(s) ...")
        self.model_dmgs = [self._load_model(model_name, "dmg").to(self.device).eval() for model_name in dmg_models]
        print("All models loaded.")

        # ── Load normalization stats ───────────────────────────────────────────
        # All models have the same stats automatically loaded form HF, we take the first one
        self._stats = self.model_locs[0].normalization_stats
        print("Stats loaded from model config.")

    def run_inference(
        self,
        fp_s2_pre: str | Path,
        fp_s2_post: str | Path,
        fp_s1_pre: str | Path,
        fp_s1_post: str | Path,
        output_fp: str | Path = None,
        geo: Geometry = None,
    ):
        """
        Run damage assessment inference on Sentinel images.

        All Sentinel images must have been preprocessed at 4m resolution. If a output path is given, the damage map
        will be saved as a GeoTIFF. Optionally, the output can be clipped to a provided geometry when saved by passing
        a shapely geometry in `geo`.

        Args:
            fp_s2_pre (str | Path): Path to Sentinel-2 pre-event image.
            fp_s2_post (str | Path): Path to Sentinel-2 post-event image.
            fp_s1_pre (str | Path): Path to Sentinel-1 pre-event image.
            fp_s1_post (str | Path): Path to Sentinel-1 post-event image.
            output_fp (str | Path): Path to output GeoTIFF that will be written. Defaults to None (no file will be written).
            geo (Geometry, optional): If provided, the output is clipped to this geometry when saved. Defaults to None.
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
        ds_s1_pre = SingleRasterDataset(str(fp_s1_pre), all_bands=S1_BANDS, transforms=NormalizeSentinel("s1", self._stats))
        ds_s1_post = SingleRasterDataset(str(fp_s1_post), all_bands=S1_BANDS, transforms=NormalizeSentinel("s1", self._stats))
        ds_s2_pre = SingleRasterDataset(str(fp_s2_pre), all_bands=S2_BANDS, transforms=NormalizeSentinel("s2", self._stats))
        ds_s2_post = SingleRasterDataset(str(fp_s2_post), all_bands=S2_BANDS, transforms=NormalizeSentinel("s2", self._stats))
        ds = reduce(lambda x, y: x & y, [ds_s1_pre, ds_s2_pre, ds_s1_post, ds_s2_post])

        # Create a grid sampler and dataloader for inference (with overlap to avoid tile boundary artefacts)
        stride = self.patch_size - 2 * self.padding
        sampler = GridGeoSampler(ds, size=self.patch_size, stride=stride)
        dataloader = tdata.DataLoader(ds, sampler=sampler, batch_size=4, num_workers=4, collate_fn=stack_samples)
        print(f"Dataloader: {len(dataloader)} batches")

        # Prepare the output array with correct georeferencing info
        output, (offset_h, offset_v), transform = prepare_output(fp_s2_pre, bbox=None, verbose=1, n_channels=1, dtype=np.uint8)

        # Inference (predict -> ensmeble -> postprocess -> stitch into output)
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader)):
                bboxes = batch["bounds"]
                img = batch["image"].to(self.device)

                # Ensemble forward pass
                logits_loc = self._forward_ensemble(img, self.model_locs, "loc")
                logits_dmg = self._forward_ensemble(img, self.model_dmgs, "dmg")

                # From logits to localization mask and damage classes
                preds_loc = (torch.sigmoid(logits_loc) > 0.5).long().squeeze(1)
                preds_dmg = torch.argmax(logits_dmg[:, 1:], dim=1) + 1
                preds_dmg[preds_loc == 0] = 0
                preds = preds_dmg.cpu().numpy()

                # Stitch batch predictions into the output raster
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

    def _load_model(self, model_name: str, loc_dmg: str) -> torch.nn.Module:
        """Try to load the model as UNetWithInputSkip first, then fall back to SiameseUnetLateFusion if that fails (for backward compatibility)."""
        assert loc_dmg in ["loc", "dmg"], f"loc_dmg must be 'loc' or 'dmg', got {loc_dmg}"
        for model_class, model_type in zip([UNetWithInputSkip, SiameseUnetLateFusion], ["unet", "siamese"]):  # noqa B905
            try:
                model = model_class.from_pretrained(pretrained_model_name_or_path=model_name)
                self.__setattr__(f"model_{loc_dmg}", model_type)
                return model
            except Exception:
                # print(f"Error occurred while loading model {model_name} with {model_type}: {e}")
                continue
        else:
            raise ValueError("Could not load the model with either UNetWithInputSkip or SiameseUnetLateFusion architectures.")

    def _forward_ensemble(self, img: torch.Tensor, models: list[nn.Module], loc_dmg: str) -> torch.Tensor:
        """Forward pass through the ensemble of models and average the predictions."""
        assert loc_dmg in ["loc", "dmg"], f"loc_dmg must be 'loc' or 'dmg', got {loc_dmg}"
        model_name = self.__getattribute__(f"model_{loc_dmg}")
        if model_name == "unet":
            logits = [model(img) for model in models]
            return torch.stack(logits).mean(dim=0)
        elif model_name == "siamese":
            img_pre = img[:, :14]
            img_post = img[:, 14:]
            logits = [model(img_pre, img_post) for model in models]
            return torch.stack(logits).mean(dim=0)
        else:
            raise ValueError(f"Unknown model type: {model_name}")


class NormalizeSentinel:
    """Normalise a torchgeo sample using per-sensor 1st/99th percentile stats."""

    def __init__(self, sensor: str, stats: dict):
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
    output_fp = folder / "output_from_hub_final.tif"

    infer = InferenceFromHub()
    infer.run_inference(fp_s2_pre, fp_s2_post, fp_s1_pre, fp_s1_post, output_fp)
