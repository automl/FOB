"""
Convenience functions for warming up lr schedulers.
"""
import math
from typing import Any, Type
import torch
from torch.optim.lr_scheduler import LinearLR, LRScheduler, SequentialLR

from pytorch_fob.engine.configs import OptimizerConfig
from pytorch_fob.engine.utils import log_info
from pytorch_fob.optimizers.lr_schedulers.schedulers import IncreasingCosineAnnealingLR


def warmup_split(max_steps: int, warmup_factor: float) -> tuple[int, int]:
    warmup_steps = int(math.ceil(max_steps * warmup_factor))
    return warmup_steps, max(max_steps - warmup_steps, 1)


def warmup_split_from_config(config: OptimizerConfig) -> tuple[int, int]:
    if config.lr_scheduler.warmup_steps is not None:
        warmup_steps = config.lr_scheduler.warmup_steps
        return warmup_steps, config.max_steps - warmup_steps
    elif config.lr_scheduler.warmup_factor is not None:
        return warmup_split(config.max_steps, config.lr_scheduler.warmup_factor)
    else:
        raise ValueError("Either 'warmup_steps' or 'warmup_factor' should be specified.")


def decay_steps_from_config(config: OptimizerConfig) -> int:
    if config.lr_scheduler.decay_steps is not None:
        return config.lr_scheduler.decay_steps
    elif config.lr_scheduler.decay_factor is not None:
        return int(math.ceil(config.max_steps * config.lr_scheduler.decay_factor))
    else:
        raise ValueError("Either 'decay_steps' or 'decay_factor' should be specified.")


def linear_warmup(
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    warmup_steps: int,
    scheduler: Type[LRScheduler],
    scheduler_kwargs: dict[str, Any]
) -> SequentialLR | LRScheduler:
    """
    Apply a linear warmup to the given optimizer using the specified scheduler.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to apply the warmup to.
        max_steps (int): The maximum number of training steps.
        warmup_steps (int): The number of warmup steps.
        scheduler (Type[LRScheduler]): The scheduler to use after the warmup.
        scheduler_kwargs (dict[str, Any]): Additional keyword arguments for the scheduler.

    Returns:
        SequentialLR | LRScheduler: The sequential LR scheduler combining the warmup and the actual scheduler.

    """
    return _warmup(optimizer, max_steps, warmup_steps, scheduler, scheduler_kwargs, warmup_strategy="linear")


def cosine_warmup(
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    warmup_steps: int,
    scheduler: Type[LRScheduler],
    scheduler_kwargs: dict[str, Any]
) -> SequentialLR | LRScheduler:
    """
    Apply a cosine warmup to the given optimizer using the specified scheduler.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to apply the warmup to.
        max_steps (int): The maximum number of training steps.
        warmup_steps (int): The number of warmup steps.
        scheduler (Type[LRScheduler]): The scheduler to use after the warmup.
        scheduler_kwargs (dict[str, Any]): Additional keyword arguments for the scheduler.

    Returns:
        SequentialLR | LRScheduler: The sequential LR scheduler combining the warmup and the actual scheduler.

    """
    return _warmup(optimizer, max_steps, warmup_steps, scheduler, scheduler_kwargs, warmup_strategy="cosine")


def _warmup(
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    warmup_steps: int,
    scheduler: Type[LRScheduler],
    scheduler_kwargs: dict[str, Any],
    warmup_strategy: str,
) -> SequentialLR | LRScheduler:
    if max_steps < 1:
        raise ValueError("max steps should be at least 1!")
    if warmup_steps == 0:
        log_info(f"warmup = 0: using {scheduler} only")
        return scheduler(optimizer, **scheduler_kwargs)
    if warmup_strategy == "linear":
        warmup_scheduler = LinearLR(
            optimizer, start_factor=1e-10, end_factor=1., total_iters=warmup_steps)
    elif warmup_strategy == "cosine":
        warmup_scheduler = IncreasingCosineAnnealingLR(
            optimizer, T_max=warmup_steps)
    else:
        raise ValueError(f"Unknown warmup strategy: {warmup_strategy}")
    actual_scheduler = scheduler(optimizer, **scheduler_kwargs)
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, actual_scheduler],
        milestones=[warmup_steps],
    )
