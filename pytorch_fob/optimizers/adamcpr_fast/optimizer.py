"""
Fast implementation of Constrained Parameter Regularization proposed in: https://arxiv.org/pdf/2311.09058v1.pdf
"""

import math
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from pytorch_fob.engine.configs import OptimizerConfig
from pytorch_fob.engine.utils import log_info
from pytorch_fob.engine.parameter_groups import GroupedModel
from pytorch_fob.optimizers.adamcpr_fast.adam_cpr_fast import AdamCPRfast


def warmup_steps(total_steps: int, warmup_factor: float) -> int:
    if total_steps < 1:
        raise ValueError("total steps should be at least 1!")
    return math.ceil(warmup_factor * total_steps)


def cosine_warmup(
        total_steps: int,
        warmup_steps: int,
        eta_min: float,
        optimizer
        ) -> SequentialLR | CosineAnnealingLR:
    if warmup_steps == 0:
        log_info("warmup = 0: using CosineAnnealingLR only")
        return CosineAnnealingLR(optimizer, T_max=total_steps)
    warmup = LinearLR(
        optimizer, start_factor=1e-10, end_factor=1., total_iters=warmup_steps)
    cosine_steps = max(total_steps - warmup_steps, 1)
    cosine_decay = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[warmup, cosine_decay], milestones=[warmup_steps])


def configure_optimizers(model: GroupedModel, config: OptimizerConfig) -> OptimizerLRScheduler:
    lr = config.learning_rate
    weight_decay = 1.0  # only 1 and 0 are valid

    parameter_groups = model.grouped_parameters(lr=lr, weight_decay=weight_decay)

    step_hint = config.max_steps if config.max_steps else config.max_epochs
    lr_warmup_steps = warmup_steps(step_hint, config.warmup_factor)
    if config.kappa_init_method == "warm_start_factor":
        kappa_init_param = int(lr_warmup_steps * config.kappa_init_param)
        kappa_init_method = "warm_start"
    else:
        if config.kappa_init_method == "inflection_point":
            kappa_init_param = lr_warmup_steps
        else:
            kappa_init_param = config.kappa_init_param
        kappa_init_method = config.kappa_init_method

    optimizer = AdamCPRfast(
        params=parameter_groups,
        lr=lr,
        betas=(1 - config.one_minus_beta1, config.beta2),
        eps=config.epsilon,
        kappa_init_param=kappa_init_param,
        kappa_init_method=kappa_init_method,
        reg_function=config.reg_function,
        kappa_update=config.kappa_update
    )
    scheduler = cosine_warmup(step_hint, lr_warmup_steps, config.eta_min_factor * lr, optimizer)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": config.lr_interval
        }
    }
