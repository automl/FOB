"""
Additional LR schedulers
"""
import math
import warnings
import torch
from torch.optim.lr_scheduler import LinearLR, LRScheduler, SequentialLR


class _CosineAnnealingLR(LRScheduler):
    """
    Same as torch.optim.lr_scheduler.CosineAnnealingLR, except `eta_min` is replaced by `eta_min_factor`.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min_factor (float): Minimum learning rate as a factor of the initial learning rate. Default: 0.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If `True`, prints a message to stdout for each update. Default: `False`.
    """

    def __init__(self, optimizer, T_max, eta_min_factor=0.0, last_epoch=-1, verbose=False):
        self.T_max = T_max
        self.eta_min_factor = eta_min_factor
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        elif self._step_count == 1 and self.last_epoch > 0:
            return [self.eta_min_factor * base_lr +
                    base_lr * (1 - self.eta_min_factor) *
                    (1 + math.cos((self.last_epoch) * math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + base_lr * (1 - self.eta_min_factor) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (group['lr'] - self.eta_min_factor * base_lr) + self.eta_min_factor * base_lr
                for base_lr, group in
                zip(self.base_lrs, self.optimizer.param_groups)]

    def _get_closed_form_lr(self):
        return [self.eta_min_factor * base_lr +
                base_lr * (1 - self.eta_min_factor) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]


class IncreasingCosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, T_max, eta_min_factor=0.0, last_epoch=-1, verbose=False):
        self.T_max = T_max
        self.eta_min_factor = eta_min_factor
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [
            self.eta_min_factor * base_lr +
            (base_lr - self.eta_min_factor * base_lr) *
            (1 - math.cos(math.pi * self.last_epoch / self.T_max)) / 2
        for base_lr in self.base_lrs]


class IdentityLR(LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


def wsd_scheduler(
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    warmup_steps: int,
    decay_steps: int,
    eta_min_factor: float = 0.0,
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
        eta_min_factor (float, optional): The minimum learning rate given as a factor of the initial learning rate. Defaults to zero.
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
                optimizer, T_max=warmup_steps, eta_min_factor=eta_min_factor)
        else:
            raise ValueError(f"Unknown warmup strategy: {warmup_strategy}")
    else:
        warmup_scheduler = None
    if decay_steps > 0:
        if decay_strategy == "linear":
            decay_scheduler = LinearLR(
                optimizer, start_factor=1., end_factor=eta_min_factor, total_iters=decay_steps)
        elif decay_strategy == "cosine":
            decay_scheduler = _CosineAnnealingLR(
                optimizer, T_max=decay_steps, eta_min_factor=eta_min_factor)
        else:
            raise ValueError(f"Unknown decay strategy: {decay_strategy}")
    else:
        decay_scheduler = None
    constant_steps = max_steps - warmup_steps - decay_steps
    constant_scheduler = IdentityLR(optimizer) if constant_steps > 0 else None
    schedulers = [s for s in [warmup_scheduler, constant_scheduler, decay_scheduler] if s is not None]
    if len(schedulers) == 1:
        return schedulers[0]
    return SequentialLR(
        optimizer,
        schedulers=schedulers,
        milestones=[warmup_steps, max_steps - decay_steps],
    )
