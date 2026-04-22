"""Trainer class for xBD-S12, containing the full training/validation/testing logic, as well as checkpointing and logging."""

import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.optim as toptim
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from src.constants import LOGS_PATH
from src.models.factory import model_factory
from src.training.dataloaders import get_dataloaders
from src.training.losses import loss_factory
from src.training.metrics import xBDS12Metrics
from src.training.scheduler import scheduler_factory
from src.training.utils import seed_everything, unbind_samples
from src.utils.time import print_sec


class Trainer:
    """Main trainer class for xBD-S12."""

    def __init__(
        self,
        run_name: str,
        task: str = "multiclass",
        # Data parameters
        which_split: str = "event",
        modalities: list | None = None,  # default is ['s1', 's2']
        data_kwargs: dict | None = None,
        # Model parameters
        model_name: str = "unet",
        model_kwargs: dict | None = None,
        # Training parameters
        max_epochs: int = 40,
        batch_size: int = 16,
        early_stopping: int | None = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        use_transforms: bool = True,
        sampler_train: str | None = "weighted",
        # Scheduler parameters
        scheduler: str | None = None,
        scheduler_kwargs: dict | None = None,
        # Loss parameters
        cfg_loss: dict | None = None,
        # Logging parameters
        logger: str = "wandb",
        wandb_project: str = "xbd-s12",
        wandb_tags: str | list | None = None,
        plot_per_train: int = 0,
        plot_per_valid: int = 0,
        plot_per_test: int = 0,
        # Other parameters
        extra_buffers_for_evaluation: int | list | None = None,
        num_workers: int = 8,
        debug: bool = False,
        seed: int = 42,
        cfg_to_save: DictConfig | None = None,
    ):
        """Initialize the trainer."""

        seed_everything(seed)

        data_kwargs = none_to_dict(data_kwargs)
        model_kwargs = none_to_dict(model_kwargs)
        scheduler_kwargs = none_to_dict(scheduler_kwargs)
        cfg_loss = none_to_dict(cfg_loss, name="ce", ignore_index=99)  # default params

        # ======= General config =======
        self.task = task
        self.debug = debug
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Training parameters
        self.max_epochs = max_epochs
        self.early_stopping = max_epochs if early_stopping is None else early_stopping

        # Checkpointing
        self.save_model_how = "both"
        self.metrics_for_best = "F1_final" if task != "localization" else "F1_loc"
        self.which_best = "max"  # keep epoch with maximum F1
        self.best_save_metric = 0.0 if self.which_best == "max" else float("inf")

        # Logging and printing
        self.metrics_print = ["F1_loc", "F1_dmg", "F1_final"] if task != "localization" else ["F1_loc"]
        self.plot_per_train = plot_per_train if plot_per_train != 0 else None
        self.plot_per_valid = plot_per_valid if plot_per_valid != 0 else None
        self.plot_per_test = plot_per_test if plot_per_test != 0 else None

        # Initalize dict with general training info
        self.info = {"epoch": 1, "iter": 0, "best_epoch": None}  # epoch starts at 1

        # ===== Dataloaders =====
        self.which_split = which_split
        modalities = ["s1", "s2"] if modalities is None else modalities
        self.modalities = [modalities] if isinstance(modalities, str) else modalities
        self.periods = ["pre", "post"]  # always pre and post

        self.dataloaders = get_dataloaders(
            modalities=self.modalities,
            task=task,
            which_split=self.which_split,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler_train=sampler_train,
            seed=seed,
            use_transforms=use_transforms,
            **data_kwargs,
        )
        print("Dataloaders created.")

        # ===== Model =====
        # Keep input_type for batch_to_logits
        self.input_type = model_kwargs.pop("input_type")
        # Figure out number of in_ and out_channels from the dataset.
        in_channels = self.dataloaders["train"].dataset.get_in_channels()
        out_channels = self.dataloaders["train"].dataset.get_out_channels()
        if self.input_type == "stacked":
            in_channels *= len(self.periods)  # *2 for pre and post
        model_kwargs["in_channels"] = in_channels
        model_kwargs["out_channels"] = out_channels

        print(f"Creating model {model_name} with {model_kwargs['in_channels']} in channels and {model_kwargs['out_channels']} out channels.")
        self.model = model_factory(model_name, **model_kwargs).to(self.device)
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model has {num_params / 1e6:.2f}M trainable parameters.")

        # ===== Metrics =====
        self.metrics = {
            "train": xBDS12Metrics(
                num_dmg_classes=2,
                ignore_index=99,
                dmg_classes_names=["intact", "damaged"],
                localization_only=task == "localization",
                extra_buffer_for_evaluation=extra_buffers_for_evaluation,
            ),
            "valid": xBDS12Metrics(
                num_dmg_classes=2,
                ignore_index=99,
                dmg_classes_names=["intact", "damaged"],
                localization_only=task == "localization",
                extra_buffer_for_evaluation=extra_buffers_for_evaluation,
            ),
            "test": xBDS12Metrics(
                num_dmg_classes=2,
                ignore_index=99,
                dmg_classes_names=["intact", "damaged"],
                localization_only=task == "localization",
                extra_buffer_for_evaluation=extra_buffers_for_evaluation,
            ),
        }

        # ===== Loss ======
        self.loss_name = cfg_loss["name"]
        self.criterion = loss_factory(cfg_loss).to(self.device)

        # ========== Optimizer ==========
        self.optimizer = toptim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if scheduler == "cosine_with_warmup":
            scheduler_kwargs["len_dataloader"] = len(self.dataloaders["train"])
        self.scheduler = scheduler_factory(self.optimizer, scheduler, **scheduler_kwargs)  # can be None
        print("Trainer initialized.")

        # ========== Logger ==========
        self.logger = logger
        self.log_folder = LOGS_PATH / run_name
        if self.logger is None:
            print("Nothing will be logged for this run")
            pass
        else:
            assert self.logger in ["wandb", "local"], "logger must be 'wandb', 'local' or None"
            self.log_folder.mkdir(parents=True, exist_ok=True)
            (self.log_folder / "images").mkdir(parents=True, exist_ok=True)
            (self.log_folder / "models").mkdir(parents=True, exist_ok=True)
            if self.logger == "wandb":
                print("Logging to Weights & Biases")
                assert wandb_project is not None, "wandb_project must be specified if use_logger is 'wandb'"
                wandb_project = "debug" if self.debug else wandb_project
                if wandb_tags is not None:
                    wandb_tags = [wandb_tags] if isinstance(wandb_tags, str) else wandb_tags
                    wandb_tags = list(wandb_tags)
                else:
                    wandb_tags = []
                if self.debug:
                    wandb_tags.append("debug")

                # add task
                wandb_tags.append(task)

                # Add data split to tags
                wandb_tags.append(which_split)

                # add only_pre vs pre+post
                which_data = "only_pre" if data_kwargs.get("only_pre_disaster", False) else "pre+post"
                wandb_tags.append(which_data)

                # add loss
                wandb_tags.append(self.loss_name)

                # add model name
                model_name_ = model_name
                if model_kwargs.get("add_skip_connection", False):
                    model_name_ += "+skip"
                wandb_tags.append(model_name_)

                wandb_tags = wandb_tags if len(wandb_tags) > 0 else None
                print(f"Run name: {run_name}")
                print(f"Run tags: {wandb_tags}")

                # Only save a subset of the config.
                cfg_to_save = (
                    cfg_to_save
                    if cfg_to_save is not None
                    else {
                        "learning_rate": lr,
                        "weight_decay": weight_decay,
                        "batch_size": batch_size,
                        "sampler_train": sampler_train,
                        "modalities": modalities,
                        "model": model_name,
                        "total_params": num_params,
                        "seed": seed,
                        "task": task,
                        "which_split": which_split,
                        **cfg_loss,
                    }
                )
                cfg_to_save = OmegaConf.to_container(cfg_to_save, resolve=True) if isinstance(cfg_to_save, DictConfig) else cfg_to_save
                wandb.init(project=wandb_project, name=run_name, dir=str(self.log_folder), tags=wandb_tags, config=cfg_to_save)
            else:
                print(f"Saving images and models locally to {self.log_folder}")

    def batch_to_logits(self, batch: dict) -> torch.Tensor:
        """Given a batch from the dataloader, prepare the input and do a forward pass to get the logits.

        If input_type is "stacked", all images are stacked in the channel dimension. The order is always alphabetical and pre then post.
        For instance, [s1_pre, s2_pre, s1_post, s2_post] if modalities are ["s1", "s2"] and periods are ["pre", "post"].
        If input_type is "pre-post", the pre and post images are stacked separately in the channel dimension.
        The order of modalities is always alphabetical. For instance, x_pre = [s1_pre, s2_pre] and x_post = [s1_post, s2_post]

        Args:
            batch (dict): a batch from the dataloader, containing at least the keys "images" and "labels".
            The "images" value is itself a dict with keys like "s1_pre", "s1_post", etc.

        Returns:
            torch.Tensor: the logits output by the model for this batch, of shape [B, 3, H, W] (0=building, 1=intact, 2=damaged)
        """

        if self.input_type == "stacked":
            channel_dim = 1  # input is BxCxHxW
            # (always sort(modalities) for consistency) and pre then post"""
            mod = sorted(self.modalities)
            x_pre = torch.cat([batch["images"][m + "_pre"] for m in mod], dim=channel_dim)
            if "post" in self.periods:
                x_post = torch.cat([batch["images"][m + "_post"] for m in mod], dim=channel_dim)
                x = torch.cat([x_pre, x_post], dim=channel_dim)
            else:
                x = x_pre
            x = x.to(self.device)  # [B, C, H, W]

            # Forward pass
            logits = self.model(x)

        elif self.input_type == "pre-post":
            # All pre images are stacked in the channel dimension, same for post
            mod = sorted(self.modalities)
            x_pre = torch.cat([batch["images"][m + "_pre"] for m in mod], dim=1).to(self.device)
            x_post = torch.cat([batch["images"][m + "_post"] for m in mod], dim=1).to(self.device)

            # Forward pass
            logits = self.model(x_pre, x_post)

        else:
            raise ValueError(f"input_type for model: {self.input_type} not recognized.")

        return logits

    def logits_to_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the loss given the logits and the labels.

        Args:
            logits (torch.Tensor): the raw output of the model, of shape [B, 3, H, W] (0=building, 1=intact, 2=damaged)
            labels (torch.Tensor): the ground truth labels, of shape [B, H, W]

        Returns:
            torch.Tensor: the computed loss
        """

        loss = self.criterion(logits, labels)
        return loss

    def logits_to_preds(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Generate the final predictions for the metrics from the logits.

        Args:
            logits (torch.Tensor): the raw output of the model, of shape [B, 3, H, W] (0=building, 1=intact, 2=damaged)

        Returns:
            tuple: the predicted localization and damage labels
            - preds_loc (torch.Tensor): predicted building localization, of shape [B, H, W], binary (0=background, 1=building)
            - preds_dmg (torch.Tensor or None): predicted damage level, of shape [B, H, W], with values 0=background, 1=intact, 2=damaged.
                None if task is "localization".
        """

        if self.task == "localization":
            # first channel predicts building
            preds_loc = torch.sigmoid(logits[:, 0]) > 0.5
            preds_dmg = None
        else:
            # first channel predicts background (so building if <0.5)
            preds_loc = torch.sigmoid(logits[:, 0]) < 0.5
            preds_dmg = torch.argmax(logits[:, 1:], dim=1) + 1

        return preds_loc, preds_dmg

    def train(self):
        """Full training loop across epochs."""

        print(f"Starting training for a maximum of {self.max_epochs} epochs.")

        for epoch in (pbar := tqdm(range(self.info["epoch"], self.max_epochs + 1), leave=True)):
            pbar.set_description(f"Epoch [{epoch}/{self.max_epochs}]")

            if self.debug and epoch == 2:
                print("\nDebug mode: stopping after 2 epochs.")
                break

            time_epoch_start = time.time()

            # Train and log metrics
            loss_train_e = self.train_epoch()
            d_metrics_train = self.metrics["train"].compute()
            d_metrics_train["loss_epoch"] = loss_train_e
            if self.logger == "wandb":
                wandb.log({f"train/{k}": v for k, v in d_metrics_train.items()}, step=self.info["iter"])
            self.metrics["train"].reset()

            # Validate
            if "valid" in self.dataloaders:
                d_metrics_val = self.validate()
                loss_val_e = d_metrics_val["loss_epoch"]  # for scheduler and printing

            # Time epoch
            time_epoch = time.time() - time_epoch_start
            if self.logger == "wandb":
                wandb.log({"epoch_time": time_epoch, "epoch": epoch}, step=self.info["iter"])

            # Print end of epoch description
            txt_end = f" Epoch [{epoch}/{self.max_epochs}] completed in {print_sec(time_epoch)}. "
            txt_end += f"Training loss: {loss_train_e:.4f}. "
            if self.metrics_print is not None:
                txt_end += " ".join([f"{k}={d_metrics_train[k]:.2f}" for k in self.metrics_print])
            if "valid" in self.dataloaders:
                txt_end += f" Validation loss: {loss_val_e:.4f}. "
                if self.metrics_print is not None:
                    txt_end += " ".join([f"{k}={d_metrics_val[k]:.2f}" for k in self.metrics_print])
            if self.scheduler is not None:
                current_lr = self.scheduler.get_last_lr()[0]
                txt_end += f" LR: {current_lr:.2e}. "
            print(txt_end)

            # save model every epoch
            if self.save_model_how in ["last", "both"]:
                self.save_model("last")

            # Early stopping
            # TODO: improve
            if self.info["best_epoch"] is not None and self.info["epoch"] - self.info["best_epoch"] > self.early_stopping:
                print(f"Early stopping at epoch {self.info['epoch']}.")
                break

            self.info["epoch"] += 1

    def train_epoch(self) -> float:
        """Training logic for one epoch."""

        # set model to training mode
        self.model.train()

        dataloader = self.dataloaders["train"]
        epoch_loss = 0.0
        skipped_batch = 0

        pbar = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
        for i, batch in pbar:
            if self.debug and i > 40:
                print("\nDebug mode: stopping after 40 batches.")
                break

            # Predict
            logits = self.batch_to_logits(batch)

            # Prepare labels
            labels = batch["labels"].to(self.device)

            # Compute loss
            loss = self.logits_to_loss(logits, labels)

            epoch_loss += loss.item()
            pbar.set_description(f"Training Loss: {loss.item():.4f}", refresh=False)

            # Sanity check: if the loss does not require grad, skip the batch (for instance if full batch is ignored)
            # Should never happen but just in case.
            if not loss.requires_grad:
                print("Skipping batch with no gradient.")
                skipped_batch += 1
                continue

            # Compute metrics
            preds_loc, preds_dmg = self.logits_to_preds(logits)
            self.metrics["train"].update(labels, preds_loc, preds_dmg)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()

            # Compute gradient norm for logging
            grad_norm = None
            if self.info["iter"] % 10 == 0:
                # Use clip_grad_norm_ to keep track of the gradient norm
                max_norm_for_clipping = float("inf")  # Set to infinity to effectively disable clipping and just compute the norm
                grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=max_norm_for_clipping)

            # optimizer step
            self.optimizer.step()

            # scheduler step
            if self.scheduler:
                self.scheduler.step()

            # Plot
            if self.plot_per_train is not None and ((i + 1) % (len(dataloader) // self.plot_per_train)) == 0 and self.logger is not None:
                batch["predictions"] = preds_loc.long() if self.task == "localization" else preds_loc.long() * preds_dmg
                fig = self.plot(batch)
                # saving locally
                fig.savefig(self.log_folder / "images" / f"train_batch{i}_e{self.info['epoch']}.png")
                if self.logger == "wandb":
                    wandb.log({f"images/train_batch_{i}": wandb.Image(fig)}, step=self.info["iter"])
                plt.close(fig)

            # log and update info
            # if i % self.logstep_train == 0:
            if self.logger == "wandb":
                data_to_log = {"train/loss": loss.item()}
                if grad_norm is not None:
                    data_to_log["train/grad_norm"] = grad_norm.item()
                if self.scheduler:
                    data_to_log["train/lr"] = self.scheduler.get_last_lr()[0]
                wandb.log(data_to_log, step=self.info["iter"])
            self.info["iter"] += 1

        # Average Loss for entire epoch
        epoch_loss_e = epoch_loss / len(dataloader)

        if skipped_batch > 0:
            print(f"Skipped {skipped_batch} batches with no gradient during this epoch.")
        return epoch_loss_e

    def validate(self) -> dict:
        """Full validation logic (one epoch + metrics and checkpointing)."""
        loss_val_e = self.val_epoch()
        d_metrics_val = self.metrics["valid"].compute()
        d_metrics_val["loss_epoch"] = loss_val_e
        if self.logger == "wandb":
            wandb.log({f"valid/{k}": v for k, v in d_metrics_val.items()}, step=self.info["iter"])
        self.metrics["valid"].reset()

        # Save model
        if self.save_model_how in ["best", "both"]:
            self.check_and_save_best_model(d_metrics_val)
        return d_metrics_val

    def val_epoch(self) -> float:
        """Validation logic for one epoch."""

        # set model to evaluation mode
        self.model.eval()

        dataloader = self.dataloaders["valid"]
        epoch_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
            for i, batch in pbar:
                if self.debug and i > 40:
                    print("\nDebug mode: stopping after 40 batches.")
                    break

                # Predict
                logits = self.batch_to_logits(batch)

                # Prepare labels
                labels = batch["labels"].to(self.device)

                # Compute loss
                loss = self.logits_to_loss(logits, labels)
                epoch_loss += loss.item()
                pbar.set_description(f"Valid Loss: {loss.item():.4f}", refresh=False)

                # Compute metrics
                preds_loc, preds_dmg = self.logits_to_preds(logits)
                self.metrics["valid"].update(labels, preds_loc, preds_dmg)

                # Plot
                if self.plot_per_valid is not None and ((i + 1) % (len(dataloader) // self.plot_per_valid)) == 0 and self.logger is not None:
                    batch["predictions"] = preds_loc.long() if self.task == "localization" else preds_loc.long() * preds_dmg
                    fig = self.plot(batch)
                    fig.savefig(self.log_folder / "images" / f"valid_batch{i}_e{self.info['epoch']}.png")
                    if self.logger == "wandb":
                        wandb.log({f"images/valid_batch_{i}": wandb.Image(fig)}, step=self.info["iter"])
                    plt.close(fig)

        # Average Loss for entire epoch
        epoch_loss_e = epoch_loss / len(dataloader)
        return epoch_loss_e

    def test(self, ckpt: str = "best") -> dict | None:
        """Test the model on the test set using the specified checkpoint (either "best" or "last").

        If no test set is available, skip testing.
        """

        if "test" not in self.dataloaders or self.dataloaders["test"] is None:
            print("No test set in the dataloaders. Skipping test.")
            return None

        if ckpt in ["best", "last"]:
            fallback = "last" if ckpt == "best" else None
            for prefix in [ckpt, fallback]:
                if prefix is None:
                    break

                try:
                    ckpt = self.find_ckpt_path(prefix=ckpt)
                    if prefix == fallback:
                        print(f"Checkpoint 'best' not found, using 'last' instead: {ckpt}.")
                    break
                except Exception as e:
                    print(e)
                    continue
            else:
                print("No checkpoint found for this run...")
                return None

        self.resume(path=ckpt, load_optimizer=False)

        # set model to evaluation mode
        self.model.eval()

        dataloader = self.dataloaders["test"]
        epoch_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
            for i, batch in pbar:
                if self.debug and i > 50:
                    print("\nDebug mode: stopping after 50 batches.")
                    break

                # Predict
                logits = self.batch_to_logits(batch)

                # Prepare labels
                labels = batch["labels"].to(self.device)

                # Compute loss
                loss = self.logits_to_loss(logits, labels)
                epoch_loss += loss.item()
                pbar.set_description(f"Test Loss: {loss.item():.4f}", refresh=False)

                # Compute metrics
                preds_loc, preds_dmg = self.logits_to_preds(logits)
                self.metrics["test"].update(labels, preds_loc, preds_dmg)

                # Plot
                if self.plot_per_test is not None and ((i + 1) % (len(dataloader) // self.plot_per_test)) == 0 and self.logger is not None:
                    batch["predictions"] = preds_loc.long() if self.task == "localization" else preds_loc.long() * preds_dmg
                    fig = self.plot(batch)
                    fig.savefig(self.log_folder / "images" / f"test_batch{i}_e{self.info['epoch']}.png")
                    if self.logger == "wandb":
                        wandb.log({f"images/test_batch_{i}": wandb.Image(fig)}, step=self.info["iter"])
                    plt.close(fig)

        # Average loss for entire test set
        epoch_loss_e = epoch_loss / len(dataloader)
        d_metrics_test = self.metrics["test"].compute()
        d_metrics_test["loss_epoch"] = epoch_loss_e
        if self.logger == "wandb":
            wandb.log({f"test/{k}": v for k, v in d_metrics_test.items()}, step=self.info["iter"])
        self.metrics["test"].reset()

        print("Test results:")
        for key, value in d_metrics_test.items():
            print(f"  {key}: {value:.4f}")
        return d_metrics_test

    def check_and_save_best_model(self, val_metrics: dict) -> None:
        """Save model as 'best' if metrics is better than previously"""

        current_save_metric = val_metrics[self.metrics_for_best]
        if self.which_best == "max":
            is_best = current_save_metric > self.best_save_metric
        else:
            is_best = current_save_metric < self.best_save_metric
        if is_best:
            self.best_save_metric = current_save_metric
            self.save_model("best")
            self.info["best_epoch"] = self.info["epoch"]

        if self.logger == "wandb":
            # log at every epoch to visualize steps
            wandb.log({f"best_{self.metrics_for_best}": float(self.best_save_metric), "best_epoch": self.info["best_epoch"]}, step=self.info["iter"])

    def save_model(self, prefix: str) -> None:
        """Save model checkpoint and everything else needed to resume jobs"""
        if self.logger is None:
            return

        torch.save(
            {
                "model": self.model.state_dict(),
                "epoch": self.info["epoch"],
                "iter": self.info["iter"],
                "optimizer": self.optimizer.state_dict(),
                "scheduler": (self.scheduler.state_dict() if self.scheduler is not None else None),
            },
            self.log_folder / "models" / f"{prefix}_checkpoint.pth",
        )

    def find_ckpt_path(self, prefix: str = "best") -> str:
        """Find the checkpoint path for the given prefix ('best' or 'last')"""

        assert prefix in ["best", "last"], f"which must be 'best' or 'last', not {prefix}"
        ckpt_path = self.log_folder / "models" / f"{prefix}_checkpoint.pth"
        assert ckpt_path.exists(), f"Checkpoint {ckpt_path} does not exist."
        return ckpt_path

    def resume(self, path: str, load_optimizer=True) -> None:
        """Resume training from checkpoint"""

        if path in ["best", "last"]:
            path = self.find_ckpt_path(prefix=path)

        assert path is not None, "No path to resume from."
        assert Path(path).exists(), f"Path {path} does not exist."

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            ckpt = torch.load(path)
        self.model.load_state_dict(ckpt["model"])
        if load_optimizer:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt["scheduler"] is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        else:
            self.scheduler = None

        # self.info["epoch"] = ckpt["epoch"]
        # self.max_epochs += self.info["epoch"]  # add the epochs already done
        # self.info["iter"] = ckpt["iter"]
        print(f"Checkpoint loaded from {path}. (from epoch {ckpt['epoch']}).")

    def plot(self, batch, skip_logger_check: bool = False) -> plt.Figure | None:
        """Plot a batch of images with predictions and labels"""

        if self.logger is None and not skip_logger_check:
            return None

        samples = unbind_samples(batch)

        n_imgs = len(self.modalities) * len(self.periods) + 1  # (pre and post) + labels
        if "predictions" in batch:
            n_imgs += 1  # add predictions
        n_figs = min(5, len(samples))  # max 5 samples
        fig, axs = plt.subplots(n_figs, n_imgs, figsize=(3 * n_imgs, 3 * n_figs))
        if n_figs == 1:
            axs = axs[None, :]
        for i in range(n_figs):
            self.dataloaders["train"].dataset.plot(samples[i], axs=axs[i], show=False, add_uid_as_ylabel=True)
        return fig

    def finish(self) -> None:
        """Properly close the wandb logger if used."""
        if self.logger == "wandb":
            wandb.finish()


def none_to_dict(d: dict | None, **kwargs):
    """Utility function to convert None to empty dict, for easier handling of optional configs."""
    if d is None:
        d = {}
    d.update(kwargs)
    return d
