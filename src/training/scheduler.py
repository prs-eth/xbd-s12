"""Scheduler factory."""

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def scheduler_factory(optimizer: torch.optim.Optimizer, scheduler_name: str | None, **scheduler_kwargs):
    """Create the learning rate scheduler from the config."""

    if scheduler_name == "cosine_with_warmup":
        assert all(k in scheduler_kwargs for k in ["warmup_epochs", "total_epochs"]), "Missing required scheduler_kwargs keys"
        assert "len_dataloader" in scheduler_kwargs, "Missing 'len_dataloader' in scheduler_kwargs"
        warmup_steps = scheduler_kwargs["warmup_epochs"] * scheduler_kwargs["len_dataloader"]
        total_steps = scheduler_kwargs["total_epochs"] * scheduler_kwargs["len_dataloader"]
        scheduler = get_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            eta_min=scheduler_kwargs.get("eta_min", 0),
        )
        print('Using "warmup_cosine" scheduler (Linear warmup + CosineAnnealingLR)')
        print(f"with {total_steps} total steps ({scheduler_kwargs['total_epochs']} epochs)")
        print(f"including {warmup_steps} warmup steps ({scheduler_kwargs['warmup_epochs']} epochs)")
    elif scheduler_name is None:
        scheduler = None
        print("No scheduler used.")
    else:
        raise ValueError(f"Unknown scheduler name: {scheduler_name}")
    return scheduler


def get_warmup_cosine_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, eta_min: float = 0):
    """
    Creates a learning rate scheduler with linear warmup and cosine annealing decay.

    Uses PyTorch's built-in CosineAnnealingLR.

    Args:
        optimizer (torch.optim.Optimizer): the optimizer
        warmup_steps (int): number of warmup steps
        total_steps (int): total number of training steps
        eta_min (float, optional): minimum learning rate. Defaults to 0.

    Returns:
        torch.optim.lr_scheduler.SequentialLR: combined warmup and cosine annealing scheduler
    """

    # Warmup scheduler: linearly increases lr from 0 to base lr
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps)  # Start from nearly 0  # End at base lr

    # Cosine annealing scheduler: smoothly decays lr
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=eta_min)

    # Combine schedulers
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    return scheduler
