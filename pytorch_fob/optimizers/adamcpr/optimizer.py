"""
Official implementation of Constrained Parameter Regularization proposed in: https://arxiv.org/abs/2311.09058
"""

from lightning.pytorch.utilities.types import OptimizerLRScheduler
from pytorch_cpr import AdamCPR

from pytorch_fob.engine.configs import OptimizerConfig
from pytorch_fob.engine.parameter_groups import GroupedModel, intersect_parameter_dicts
from pytorch_fob.optimizers.adamcpr.group_parameter import group_parameters_for_cpr_optimizer
from pytorch_fob.optimizers.lr_schedulers import get_lr_scheduler, warmup_split_from_config


def configure_optimizers(model: GroupedModel, config: OptimizerConfig) -> OptimizerLRScheduler:
    lr = config.learning_rate

    parameter_groups_model = model.grouped_parameters(
        lr=lr,
        weight_decay=None  # wd is set to zero in CPR anyways
    )

    parameter_groups_cpr = group_parameters_for_cpr_optimizer(
        model=model.model,
        regularize_embedding=config.regularize_embedding
    )

    parameter_groups = []
    for pgm in parameter_groups_model:
        for pgc in parameter_groups_cpr:
            pgmc = intersect_parameter_dicts(pgc, pgm)
            if pgmc is not None:
                parameter_groups.append(pgmc)

    lr_warmup_steps, _ = warmup_split_from_config(config)
    if config.kappa_init_method == "warm_start_factor":
        kappa_init_param = int(lr_warmup_steps * config.kappa_init_param)
        kappa_init_method = "warm_start"
    else:
        kappa_init_param = config.kappa_init_param
        kappa_init_method = config.kappa_init_method
    optimizer = AdamCPR(
        params=parameter_groups,
        lr=lr,
        eps=config.epsilon,
        betas=(config.beta1, config.beta2),
        kappa_init_param=kappa_init_param,
        kappa_init_method=kappa_init_method,
        reg_function=config.reg_function,
        kappa_update=config.kappa_update,
        reg_by_lr=config.reg_by_lr,
        reg_step_size=config.reg_step_size,
    )
    scheduler = get_lr_scheduler(optimizer, config)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": config.lr_scheduler.interval
        }
    }
