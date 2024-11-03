from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LRScheduler, PolynomialLR, StepLR

from pytorch_fob.engine.configs import OptimizerConfig
from pytorch_fob.engine.utils import log_warn

from .schedulers import IdentityLR, wsd_scheduler
from .warmup import (
    _warmup,
    cosine_warmup,
    decay_steps_from_config,
    linear_warmup,
    warmup_split,
    warmup_split_from_config,
)

__all__ = ["cosine_warmup", "get_lr_scheduler", "IdentityLR", "linear_warmup", "lr_schedulers_path",
           "warmup_split", "warmup_split_from_config", "wsd_scheduler"]


def lr_schedulers_path() -> Path:
    return Path(__file__).resolve().parent


def get_lr_scheduler(optimizer: torch.optim.Optimizer, config: OptimizerConfig) -> LRScheduler:
    """
    Returns an LR scheduler based on the configuration settings.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which the scheduler is created.
        config (OptimizerConfig): The configuration settings for the optimizer.

    Returns:
        LRScheduler: The learning rate scheduler based on the specified configuration.
    """
    warmup_steps, scheduler_steps = warmup_split_from_config(config)
    scheduler_name = config.lr_scheduler.scheduler
    if scheduler_name is None or scheduler_name == "identity":
        base_scheduler = IdentityLR
        scheduler_kwargs = dict()
    elif scheduler_name == "cosine":
        base_scheduler = CosineAnnealingLR
        scheduler_kwargs = dict(
            T_max=scheduler_steps,
            eta_min=config.lr_scheduler.eta_min_factor * config.learning_rate,
        )
    elif scheduler_name == "poly":
        base_scheduler = PolynomialLR
        scheduler_kwargs = dict(
            power=config.lr_scheduler.power,
            total_iters=scheduler_steps,
        )
    elif scheduler_name == "wsd":
        # no need for additional warmup
        return wsd_scheduler(
            optimizer,
            max_steps=config.max_steps,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps_from_config(config),
            eta_min_factor=config.lr_scheduler.eta_min_factor,
            warmup_strategy=config.lr_scheduler.warmup_strategy,
            decay_strategy=config.lr_scheduler.decay_strategy,
        )
    elif scheduler_name == "stepwise":
        if config.lr_scheduler.lr_interval != "epoch":
            log_warn(
                f"Using stepwise scheduler with interval {config.lr_scheduler.lr_interval}. Make sure to set the step size accordingly."
            )
        base_scheduler = StepLR
        scheduler_kwargs = dict(
            step_size=config.lr_scheduler.step_size,
            gamma=config.lr_scheduler.gamma,
        )
    elif scheduler_name == "exponential":
        base_scheduler = ExponentialLR
        scheduler_kwargs = dict(
            gamma=config.lr_scheduler.gamma,
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    return _warmup(
        optimizer,
        max_steps=config.max_steps,
        warmup_steps=warmup_steps,
        scheduler=base_scheduler,
        scheduler_kwargs=scheduler_kwargs,
        warmup_strategy=config.lr_scheduler.warmup_strategy,
    )
