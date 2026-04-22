"""Main script to train a model on xBD-S12 from a hydra config."""

import hydra
from omegaconf import DictConfig, OmegaConf

from src.constants import HYDRA_CONFIG_PATH
from src.training.trainer import Trainer
from src.utils.time import timeit


def get_trainer_from_hydra_cfg(cfg: DictConfig) -> Trainer:
    """
    Convert the Hydra config to the arguments needed to initialize the Trainer class, and create the Trainer instance.

    Could probably be cleaner and simplified...

    Args:
        cfg: The Hydra config, see src/configs

    """

    # Parse cfg for factory functions
    data_kwargs = {k: v for k, v in cfg.data.items() if k not in ["which_split", "modalities"]}
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "name"}
    scheduler_kwargs = {k: v for k, v in cfg.scheduler.items() if k != "name"}

    return Trainer(
        run_name=cfg.run_name,
        task=cfg.task,
        # Data parameters
        which_split=cfg.data.which_split,
        modalities=cfg.data.modalities,
        data_kwargs=data_kwargs,
        # Model parameters
        model_name=cfg.model.name,
        model_kwargs=model_kwargs,
        # Training parameters
        max_epochs=cfg.training.max_epochs,
        batch_size=cfg.training.batch_size,
        early_stopping=cfg.training.early_stopping,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        use_transforms=cfg.training.use_transforms,
        sampler_train=cfg.training.sampler_train,
        # Scheduler parameters
        scheduler=cfg.scheduler.name,
        scheduler_kwargs=scheduler_kwargs,
        # Loss parameters
        cfg_loss=cfg.loss,
        # Logging parameters
        logger=cfg.logger,
        wandb_project=cfg.wandb_project,
        wandb_tags=cfg.wandb_tags,
        plot_per_train=cfg.plot_per_train,
        plot_per_valid=cfg.plot_per_valid,
        plot_per_test=cfg.plot_per_test,
        # Other
        extra_buffers_for_evaluation=cfg.extra_buffer_for_evaluation,
        num_workers=cfg.num_workers,
        debug=cfg.debug,
        seed=cfg.seed,
        cfg_to_save=cfg,  # Pass full config to save in wandb.
    )


@timeit
@hydra.main(version_base=None, config_path=str(HYDRA_CONFIG_PATH), config_name="base")
def main(cfg: DictConfig = None):
    """Launch training from a Hydra config."""

    print("Starting training...")

    # Make sure everything is resovled
    OmegaConf.resolve(cfg)

    trainer = get_trainer_from_hydra_cfg(cfg)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted. Last validation loop...")
    try:
        trainer.test()
    except Exception:
        print("Inference interrupted.")
    trainer.finish()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
