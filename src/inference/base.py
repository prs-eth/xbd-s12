import warnings
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.constants import LOGS_PATH
from src.models.factory import model_factory


class Inference:
    """Base class for inference (just load the models)"""

    def __init__(
        self,
        run_dmg: str | Path | list[str | Path] = None,
        run_loc: str | Path | list[str | Path] = None,
        verbose_model: bool = False,
    ):
        """
        Initialize the inference class by loading the localization and damage models from the provided run names.

        If multiple runs are provided, the predictions from each model will be averaged during inference.

        Args:
            run_dmg (str | Path | list[str  |  Path], optional): Path to damage model checkpoints. Defaults to None.
            run_loc (str | Path | list[str  |  Path], optional): Path to localization model checkpoints. Defaults to None.
            verbose_model (bool, optional): Verbosity level. Defaults to False.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load localization models if provided
        if run_loc is None:
            self.model_locs = []
        else:
            self.cfg_model_loc = []
            run_loc = [run_loc] if isinstance(run_loc, str) else run_loc
            self.model_locs = [self.load_model(r, verbose_model=verbose_model, loc_or_dmg="loc") for r in run_loc]

        # Load damage models
        if run_dmg is None:
            self.model_dmgs = []
        else:
            self.cfg_model_dmg = []
            run_dmg = [run_dmg] if isinstance(run_dmg, str) else run_dmg
            self.model_dmgs = [self.load_model(r, verbose_model=verbose_model, loc_or_dmg="dmg") for r in run_dmg]

    def load_model(self, run_name: str | Path, verbose_model: bool = False, loc_or_dmg: str = None) -> torch.nn.Module:
        cfg_path = LOGS_PATH / run_name / ".hydra" / "config.yaml"
        assert cfg_path.exists(), f"Config file {cfg_path} does not exist"
        cfg = OmegaConf.load(cfg_path)
        model_kwargs = {k: v for k, v in cfg.model.items() if k not in ["name", "input_type"]}
        input_type = cfg.model.input_type

        # store some config info for inferences
        cfg_model = {
            "name": cfg.model.name,
            "task": cfg.task,
            "modalities": [cfg.data.modalities] if isinstance(cfg.data.modalities, str) else cfg.data.modalities,
            "input_type": input_type,
            "periods": ["pre", "post"] if not cfg.data.get("only_pre_disaster", False) else ["pre"],
        }
        if loc_or_dmg == "loc":
            self.cfg_model_loc.append(cfg_model)
        elif loc_or_dmg == "dmg":
            self.cfg_model_dmg.append(cfg_model)

        # number of input/output channels
        in_channels = 0
        if "s2" in cfg_model["modalities"]:
            s2_bands = cfg.data.get("s2_bands", "all")
            if s2_bands == "all":
                in_channels += 12  # level-2A
            elif s2_bands == "rgb":
                in_channels += 3
            else:
                in_channels += len(s2_bands)
        if "s2_tci" in cfg_model["modalities"]:
            in_channels += 3
        if "s1" in cfg_model["modalities"]:
            s1_bands = cfg.data.get("s1_bands", "all")
            if s1_bands == "all":
                in_channels += 2
            else:
                in_channels += len(s1_bands)
        if cfg_model["input_type"] == "stacked":
            model_kwargs["in_channels"] = in_channels * len(cfg_model["periods"])  # *2 for pre and post
        else:
            model_kwargs["in_channels"] = in_channels

        # out channels
        if cfg_model["task"] == "localization":
            model_kwargs["out_channels"] = 1  # building vs no-building
        else:
            model_kwargs["out_channels"] = 3
        model = model_factory(cfg_model["name"], verbose=verbose_model, **model_kwargs)

        # Resume from checkpoint
        ckpt_path = LOGS_PATH / run_name / "models" / "best_checkpoint.pth"
        assert ckpt_path.exists(), f"Checkpoint {ckpt_path} does not exist"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        print(f"Model {run_name} loaded from {ckpt_path}")

        # To device and eval mode
        model = model.to(self.device)
        model = model.eval()
        return model

    def batch_to_logits(self, model, batch, cfg_model=None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        if cfg_model["input_type"] == "stacked":
            # All images are stacked in the channel dimension
            # (always sort(modalities) for consistency) and pre then post"""
            mod = sorted(cfg_model["modalities"])
            x_pre = torch.cat([batch["images"][m + "_pre"] for m in mod], dim=1)
            if "post" in cfg_model["periods"]:
                x_post = torch.cat([batch["images"][m + "_post"] for m in mod], dim=1)
                x = torch.cat([x_pre, x_post], dim=1)
            else:
                x = x_pre
            x = x.to(self.device)  # [B, C, H, W]

            # Forward pass
            logits = model(x)

        elif cfg_model["input_type"] == "pre-post":
            # All pre images are stacked in the channel dimension, same for post
            mod = sorted(cfg_model["modalities"])
            x_pre = torch.cat([batch["images"][m + "_pre"] for m in mod], dim=1).to(self.device)
            x_post = torch.cat([batch["images"][m + "_post"] for m in mod], dim=1).to(self.device)

            # Forward pass
            logits = model(x_pre, x_post)
        else:
            raise ValueError(f"input_type for model: {cfg_model['input_type']} not recognized.")

        return logits  # [B, 3, H, W]  # 0=building, 1=intact, 2=damaged
