"""
Constrained Parameter Regularization proposed in: https://arxiv.org/pdf/2311.09058v1.pdf
"""

import math
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning_utilities.core.rank_zero import rank_zero_info
from pytorch_cpr import CPR, cpr_group_named_parameters
from engine.configs import OptimizerConfig
from engine.parameter_groups import GroupedModel, intersect_parameter_dicts


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
        rank_zero_info("warmup = 0: using CosineAnnealingLR only")
        return CosineAnnealingLR(optimizer, T_max=total_steps)
    warmup = LinearLR(
        optimizer, start_factor=1e-10, end_factor=1., total_iters=warmup_steps)
    cosine_steps = max(total_steps - warmup_steps, 1)
    cosine_decay = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[warmup, cosine_decay], milestones=[warmup_steps])


def configure_optimizers(model: GroupedModel, config: OptimizerConfig) -> OptimizerLRScheduler:
    lr = config.learning_rate
    optimizer_args = dict(
        lr=lr,
        eps=config.epsilon,
        betas=(1.0 - config.one_minus_beta1, config.beta2),
        weight_decay=0,
        fused=False
    )

    parameter_groups_model = model.grouped_parameters(
        lr=lr,
        weight_decay=None  # wd is set to zero in CPR anyways
    )

    parameter_groups_cpr = cpr_group_named_parameters(
        model=model.model,
        optim_hps={}  # passed directly to Adam
    )

    parameter_groups = []
    for pgm in parameter_groups_model:
        for pgc in parameter_groups_cpr:
            pgmc = intersect_parameter_dicts(pgc, pgm)
            if pgmc is not None:
                parameter_groups.append(pgmc)

    base_optimizer = Adam(
        parameter_groups,
        **optimizer_args
    )
    step_hint = config.max_steps if config.max_steps else config.max_epochs
    lr_warmup_steps = warmup_steps(step_hint, config.warmup_factor)
    if config.kappa_init_method == "warm_start_factor":
        kappa_init_param = int(lr_warmup_steps * config.kappa_init_param)
        kappa_init_method = "warm_start"
    else:
        kappa_init_param = config.kappa_init_param
        kappa_init_method = config.kappa_init_method
    optimizer = CPR(
        base_optimizer,
        kappa_init_param=kappa_init_param,
        kappa_init_method=kappa_init_method,
        reg_function=config.reg_function,
        kappa_adapt=config.kappa_adapt,
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
