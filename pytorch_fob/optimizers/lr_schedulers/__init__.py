import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, PolynomialLR
from pytorch_fob.engine.configs import OptimizerConfig
from .warmup import cosine_warmup, decay_steps_from_config, linear_warmup, \
                    warmup_split, warmup_split_from_config, _warmup
from .schedulers import wsd_scheduler

__all__ = ["cosine_warmup", "get_lr_scheduler", "linear_warmup",
           "warmup_split", "warmup_split_from_config", "wsd_scheduler"]


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
    if config.lr_scheduler.scheduler == "cosine":
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
