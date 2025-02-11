from pathlib import Path
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LRScheduler, PolynomialLR, StepLR
from pytorch_fob.engine.configs import OptimizerConfig
from pytorch_fob.engine.utils import log_warn
from .warmup import cosine_warmup, decay_steps_from_config, linear_warmup, \
                    warmup_split, warmup_split_from_config, _warmup
from .schedulers import wsd_scheduler, IdentityLR

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
    if config.lr_scheduler.warmup_steps is None or config.lr_scheduler.scheduler == "identity":
        base_scheduler = IdentityLR
        scheduler_kwargs = dict()
    elif config.lr_scheduler.scheduler == "cosine":
        base_scheduler = CosineAnnealingLR
        scheduler_kwargs = dict(
            T_max=scheduler_steps,
            eta_min=config.lr_scheduler.eta_min_factor * config.learning_rate,
        )
    elif config.lr_scheduler.scheduler == "poly":
        base_scheduler = PolynomialLR
        scheduler_kwargs = dict(
            power=config.lr_scheduler.power,
            total_iters=scheduler_steps,
        )
    elif config.lr_scheduler.scheduler == "wsd":
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
    elif config.lr_scheduler.scheduler == "stepwise":
        if config.lr_scheduler.lr_interval != "epoch":
            log_warn(
                f"Using stepwise scheduler with interval {config.lr_scheduler.lr_interval}. Make sure to set the step size accordingly."
            )
        base_scheduler = StepLR
        scheduler_kwargs = dict(
            step_size=config.lr_scheduler.step_size,
            gamma=config.lr_scheduler.gamma,
        )
    elif config.lr_scheduler.scheduler == "exponential":
        base_scheduler = ExponentialLR
        scheduler_kwargs = dict(
            gamma=config.lr_scheduler.gamma,
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.lr_scheduler.scheduler}")
    return _warmup(
        optimizer,
        max_steps=config.max_steps,
        warmup_steps=warmup_steps,
        scheduler=base_scheduler,
        scheduler_kwargs=scheduler_kwargs,
        warmup_strategy=config.lr_scheduler.warmup_strategy,
    )
