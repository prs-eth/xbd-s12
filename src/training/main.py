import time
import warnings
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
import torch.optim as toptim
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from src.constants import HYDRA_CONFIG_PATH, LOGS_PATH
from src.models.factory import model_factory
from src.training.dataloaders import get_dataloaders
from src.training.losses import loss_factory
from src.training.metrics import xBDS12Metrics
from src.training.scheduler import scheduler_factory
from src.training.utils import seed_everything, unbind_samples


def get_trainer_from_cfg(cfg: DictConfig):
    """Interface between the config and the Trainer class."""

    # Parse cfg for factory functions
    data_kwargs = {k: v for k, v in cfg.data.items() if k not in ["modalities", "which_split"]}
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "name"}
    scheduler_kwargs = {k: v for k, v in cfg.scheduler.items() if k != "name"}

    return Trainer(
        run_name=cfg.run_name,
        modalities=cfg.data.modalities,
        which_split=cfg.data.get("which_split", "hafner"),
        task=cfg.get("task", "multiclass"),
        fraction_valid=cfg.data.fraction_valid,
        sampler_train=cfg.data.sampler_train,
        data_kwargs=data_kwargs,
        model_name=cfg.model.name,
        model_kwargs=model_kwargs,
        use_transforms=cfg.training.get("use_transforms", True),
        cfg_loss=cfg.loss,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        scheduler=cfg.scheduler.name,
        scheduler_kwargs=scheduler_kwargs,
        max_epochs=cfg.training.max_epochs,
        batch_size=cfg.training.batch_size,
        early_stopping=cfg.training.early_stopping,
        compute_loss_per_channel=cfg.training.get("compute_loss_per_channel", False),
        first_channel_as_building=cfg.training.get("first_channel_as_building", False),
        extra_buffers_for_evaluation=cfg.get("extra_buffer_for_evaluation", None),
        logger=cfg.logger,
        wandb_project=cfg.wandb_project,
        wandb_tags=cfg.get("wandb_tags", None),
        plot_per_train=cfg.plot_per_train,
        plot_per_valid=cfg.plot_per_valid,
        plot_per_test=cfg.plot_per_test,
        num_workers=cfg.num_workers,
        debug=cfg.debug,
        seed=cfg.seed,
        cfg_to_save=cfg,
    )


class Trainer:

    def __init__(
        self,
        run_name: str,
        task: str = "multiclass",
        extra_buffers_for_evaluation: int | list = None,
        data_kwargs: dict = {},
        model_name: str = "merlin",
        model_kwargs: dict = {},
        cfg_loss: dict = {"name": "ce"},
        scheduler: str = None,
        scheduler_kwargs: dict = {},
        max_epochs: int = 40,
        batch_size: int = 16,
        early_stopping: int = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        use_transforms: bool = True,
        sampler_train="weighted",
        logger: str = "wandb",
        wandb_project: str = "xbd-s12",
        plot_per_train: int = 0,
        plot_per_valid: int = 0,
        plot_per_test: int = 0,
        num_workers: int = 8,
        debug: bool = False,
        seed: int = 42,
    ):

        self.seed = seed
        seed_everything(self.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ======= General config =======
        self.run_name = run_name
        self.debug = debug
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.extra_buffers_for_evaluation = extra_buffers_for_evaluation
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.plot_per_train = plot_per_train
        self.plot_per_valid = plot_per_valid
        self.plot_per_test = plot_per_test
        self.metrics_print = ["F1_loc", "F1_dmg", "F1_final"] if task != "localization" else ["F1_loc"]
        self.save_model_how = "both"
        self.metrics_for_best = "F1_final" if task != "localization" else "F1_loc"
        self.which_best = "max"  # "min" or "max"
        self.best_save_metric = 0.0 if self.which_best == "max" else float("inf")
        self.info = {"epoch": 1, "iter": 0, "best_epoch": None}  # epoch starts at 1

        # ===== Data and dataloaders =====
        self.dataloaders = get_dataloaders(
            task=task,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler_train=sampler_train,
            use_transforms=use_transforms,
            **data_kwargs,
        )
        self.ds = self.dataloaders["train"].dataset  # arbitrary dataset to get properties (eg in/out channels, modalities, plotting, ...)

        # ===== Model =====
        self.model_name = model_name
        in_channels = self.ds.get_in_channels()
        out_channels = self.ds.get_out_channels()
        if self.input_type == "stacked":
            model_kwargs["in_channels"] = in_channels * len(self.ds.periods)  # *2 for pre and post
        else:
            model_kwargs["in_channels"] = in_channels
        model_kwargs["out_channels"] = out_channels
        print(f"Creating model {self.model_name} with {model_kwargs['in_channels']} in channels and {model_kwargs['out_channels']} out channels.")
        self.model = model_factory(model_name, **model_kwargs).to(self.device)
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model has {num_params / 1e6:.2f}M trainable parameters.")

        # ===== Metrics =====
        self.metrics = {
            "train": xBDS12Metrics(
                num_dmg_classes=2,
                ignore_index=99,
                dmg_classes_names=["intact", "damaged"],
                localization_only=self.task == "localization",
                extra_buffer_for_evaluation=self.extra_buffers_for_evaluation,
            ),
            "valid": xBDS12Metrics(
                num_dmg_classes=2,
                ignore_index=99,
                dmg_classes_names=["intact", "damaged"],
                localization_only=self.task == "localization",
                extra_buffer_for_evaluation=self.extra_buffers_for_evaluation,
            ),
            "test": xBDS12Metrics(
                num_dmg_classes=2,
                ignore_index=99,
                dmg_classes_names=["intact", "damaged"],
                localization_only=self.task == "localization",
                extra_buffer_for_evaluation=self.extra_buffers_for_evaluation,
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
        if self.logger is None:
            print("Nothing will be logged for this run")
            pass
        elif self.logger in ["wandb", "local"]:
            self.log_folder = LOGS_PATH / self.run_name
            self.log_folder.mkdir(parents=True, exist_ok=True)
            (self.log_folder / "images").mkdir(parents=True, exist_ok=True)
            (self.log_folder / "models").mkdir(parents=True, exist_ok=True)
            if self.logger == "wandb":
                print("Logging to Weights & Biases")
                wandb.init(project=wandb_project, name=self.run_name, dir=str(self.log_folder))
            else:
                print("Logging locally")
        else:
            raise ValueError("logger must be 'wandb', 'local' or None")

    def batch_to_logits(self, batch):

        if self.input_type == "stacked":
            # (always sort(modalities) for consistency) and pre then post"""
            mod = sorted(self.ds.modalities)
            x_pre = torch.cat([batch["images"][m + "_pre"] for m in mod], dim=1)
            if "post" in self.ds.periods:
                x_post = torch.cat([batch["images"][m + "_post"] for m in mod], dim=1)
                x = torch.cat([x_pre, x_post], dim=1)
            else:
                x = x_pre
            x = x.to(self.device)  # [B, C, H, W]

            # Forward pass
            logits = self.model(x)
        elif self.input_type == "pre-post":
            # All pre images are stacked in the channel dimension, same for post
            mod = sorted(self.ds.modalities)
            x_pre = torch.cat([batch["images"][m + "_pre"] for m in mod], dim=1).to(self.device)
            x_post = torch.cat([batch["images"][m + "_post"] for m in mod], dim=1).to(self.device)

            # Forward pass
            logits = self.model(x_pre, x_post)
        else:
            raise ValueError(f"input_type for model: {self.input_type} not recognized.")

        return logits  # eg for multiclass: [B, 3, H, W] with 0=building, 1=intact, 2=damaged

    def logits_to_preds(self, logits: torch.Tensor):

        if self.task == "localization":
            preds_loc = torch.sigmoid(logits[:, 0]) > 0.5  # first channel is building presence
            preds_dmg = None
        else:
            preds_loc = torch.sigmoid(logits[:, 0]) < 0.5  # first channel is background
            preds_dmg = torch.argmax(logits[:, 1:], dim=1) + 1

        return preds_loc, preds_dmg

    def train(self):
        """Full training/validation loop."""

        print(f"Starting training for a maximum of {self.max_epochs} epochs.")

        for epoch in (pbar := tqdm(range(self.info["epoch"], self.max_epochs + 1), leave=True)):

            pbar.set_description(f"Epoch [{epoch}/{self.max_epochs}]")

            if self.debug and epoch == 2:
                print("Debug mode: stopping after 2 epochs.")
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
            txt_end = f" Epoch [{epoch}/{self.max_epochs}] completed in {time_epoch//60:.0f}min and {time_epoch % 60:.0f}s."
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
            if self.info["best_epoch"] is not None and self.info["epoch"] - self.info["best_epoch"] > self.early_stopping:
                print(f"Early stop at epoch {self.info['epoch']}.")
                break

            self.info["epoch"] += 1

    def train_epoch(self):
        """Train for one epoch."""

        # set model to training mode
        self.model.train()

        dataloader = self.dataloaders["train"]
        epoch_loss = 0.0
        skipped_batch = 0

        pbar = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
        for i, batch in pbar:

            if self.debug and i > 40:
                print("Debug mode: stopping after 40 batches.")
                break

            # Predict
            logits = self.batch_to_logits(batch)

            # Prepare labels
            labels = batch["labels"].to(self.device)

            # Compute loss
            loss = self.criterion(logits, labels)

            epoch_loss += loss.item()
            pbar.set_description(f"Training Loss: {loss.item():.4f}", refresh=False)

            # Sanity check: if the loss does not require grad, skip the batch (for instance if full batch is ignored)
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

    def validate(self):
        """Validation step including logging and model saving."""
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

    def val_epoch(self):
        """Validation epoch."""

        # set model to evaluation mode
        self.model.eval()

        dataloader = self.dataloaders["valid"]
        epoch_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
            for i, batch in pbar:

                if self.debug and i > 40:
                    print("Debug mode: stopping after 40 batches.")
                    break

                # Predict
                logits = self.batch_to_logits(batch)

                # Prepare labels
                labels = batch["labels"].to(self.device)

                # Compute loss
                loss = self.criterion(logits, labels)
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

    def test(self, ckpt: str = "best"):
        """Test the model on the test set."""

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

    def check_and_save_best_model(self, val_metrics: dict):
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

    def save_model(self, prefix: str):
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

    def resume(self, path, load_optimizer=True):
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

        print(f"Checkpoint loaded from {path}. (from epoch {ckpt['epoch']}).")

    def plot(self, batch):
        """Plot a batch of images with predictions and labels"""

        if self.logger is None:
            return

        samples = unbind_samples(batch)
        n_imgs = self.ds.get_n_imgs(add_predictions=True)
        n_figs = min(5, len(samples))  # max 5 samples
        fig, axs = plt.subplots(n_figs, n_imgs, figsize=(3 * n_imgs, 3 * n_figs))
        if n_figs == 1:
            axs = axs[None, :]
        for i in range(n_figs):
            self.ds.plot(samples[i], axs=axs[i], show=False, add_uid_as_ylabel=True, add_titles=i == 0)
        return fig

    def finish(self):
        if self.logger == "wandb":
            wandb.finish()


@hydra.main(version_base=None, config_path=str(HYDRA_CONFIG_PATH), config_name="base")
def main(cfg: DictConfig = None):

    print("Starting training...")

    # Make sure everything is resovled
    OmegaConf.resolve(cfg)

    trainer = get_trainer_from_cfg(cfg)
    start = time.time()
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted. Last validation loop...")
    try:
        trainer.test()
    except Exception:
        print("Inference interrupted.")
    trainer.finish()

    total_time = time.time() - start
    print(f"Total time: {total_time//60:.0f}min and {total_time % 60:.0f}s.")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
