"""
Additional LR schedulers
"""
import math
import torch
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR


class IncreasingCosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 - math.cos(math.pi * self.last_epoch / self.T_max)) / 2
        for base_lr in self.base_lrs]


def wsd_scheduler(
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    warmup_steps: int,
    decay_steps: int,
    warmup_strategy: str = "linear",
    decay_strategy: str = "linear"
) -> SequentialLR | LRScheduler:
    """
    Create a Warmup-Stable-Decay (WSD) LR scheduler for an optimizer.
    Proposed in [MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies](https://arxiv.org/abs/2404.06395)

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule.
        max_steps (int): The total number of steps.
        warmup_steps (int): The number of warmup steps.
        decay_steps (int): The number of decay steps.
        warmup_strategy (str, optional): The warmup strategy. Defaults to "linear". Options: "linear", "cosine".
        decay_strategy (str, optional): The decay strategy. Defaults to "linear". Options: "linear", "cosine".

    Returns:
        Union[SequentialLR, LRScheduler]: The scheduler.

    """
    if max_steps < 1:
        raise ValueError("max steps should be at least 1!")
    if warmup_steps > 0:
        if warmup_strategy == "linear":
            warmup_scheduler = LinearLR(
                optimizer, start_factor=1e-10, end_factor=1., total_iters=warmup_steps)
        elif warmup_strategy == "cosine":
            warmup_scheduler = IncreasingCosineAnnealingLR(
                optimizer, T_max=warmup_steps)
        else:
            raise ValueError(f"Unknown warmup strategy: {warmup_strategy}")
    else:
        warmup_scheduler = None
    if decay_steps > 0:
        if decay_strategy == "linear":
            decay_scheduler = LinearLR(
                optimizer, start_factor=1., end_factor=1e-10, total_iters=decay_steps)
        elif decay_strategy == "cosine":
            decay_scheduler = CosineAnnealingLR(
                optimizer, T_max=decay_steps)
        else:
            raise ValueError(f"Unknown decay strategy: {decay_strategy}")
    else:
        decay_scheduler = None
    consant_steps = max_steps - warmup_steps - decay_steps
    if consant_steps > 0:
        constant_scheduler = ConstantLR(optimizer, total_iters=consant_steps, factor=1.0)
    else:
        constant_scheduler = None
    schedulers = [s for s in [warmup_scheduler, decay_scheduler, constant_scheduler] if s is not None]
    if len(schedulers) == 1:
        return schedulers[0]
    return SequentialLR(
        optimizer, schedulers=schedulers, milestones=[warmup_steps, max_steps - decay_steps])
