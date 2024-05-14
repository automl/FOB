import math
from typing import Any, Type
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, PolynomialLR, SequentialLR
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from pytorch_fob.engine.configs import OptimizerConfig
from pytorch_fob.engine.utils import log_info
from pytorch_fob.engine.parameter_groups import GroupedModel


def warmup_split(max_steps: int, warmup_factor: float) -> tuple[int, int]:
    warmup_steps = int(math.ceil(max_steps * warmup_factor))
    return warmup_steps, max(max_steps - warmup_steps, 1)


def linear_warmup(
        optimizer,
        max_steps: int,
        warmup_steps: int,
        scheduler: Type[CosineAnnealingLR] | Type[PolynomialLR],
        scheduler_kwargs: dict[str, Any]
    ) -> SequentialLR | CosineAnnealingLR | PolynomialLR:
    if max_steps < 1:
        raise ValueError("max steps should be at least 1!")
    if warmup_steps == 0:
        log_info(f"warmup = 0: using {scheduler} only")
        return scheduler(optimizer, **scheduler_kwargs)
    warmup = LinearLR(
        optimizer, start_factor=1e-10, end_factor=1., total_iters=warmup_steps)
    actual_scheduler = scheduler(optimizer, **scheduler_kwargs)
    return SequentialLR(
        optimizer, schedulers=[warmup, actual_scheduler], milestones=[warmup_steps])


def configure_optimizers(model: GroupedModel, config: OptimizerConfig) -> OptimizerLRScheduler:
    lr = config.learning_rate
    weight_decay = config.weight_decay
    parameter_groups = model.grouped_parameters(lr=lr, weight_decay=weight_decay)
    optimizer = AdamW(
        parameter_groups,
        lr=lr,
        eps=config.epsilon,
        betas=(1.0 - config.one_minus_beta1, config.beta2),
        weight_decay=weight_decay,
        fused=False
    )
    min_lr = config.eta_min_factor * lr
    if config.warmup_steps is not None:
        warmup_steps = config.warmup_steps
        scheduler_steps = config.max_steps - warmup_steps
    elif config.warmup_factor is not None:
        warmup_steps, scheduler_steps = warmup_split(config.max_steps, config.warmup_factor)
    else:
        raise ValueError("Either 'warmup_steps' or 'warmup_factor' should be specified.")
    if config.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR
        scheduler_kwargs = dict(
            T_max=scheduler_steps,
            eta_min=min_lr
        )
    elif config.lr_scheduler == "poly":
        scheduler = PolynomialLR
        scheduler_kwargs = dict(
            power=config.lr_power,
            total_iters=scheduler_steps
        )
    else:
        raise ValueError(f"unknown lr_scheduler: {config.lr_scheduler}")
    lr_scheduler = linear_warmup(
        optimizer,
        max_steps=config.max_steps,
        warmup_steps=warmup_steps,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": lr_scheduler,
            "interval": config.lr_interval
        }
    }
