"""
Fast implementation of Constrained Parameter Regularization proposed in: https://arxiv.org/pdf/2311.09058v1.pdf
"""

from lightning.pytorch.utilities.types import OptimizerLRScheduler
from pytorch_fob.engine.configs import OptimizerConfig
from pytorch_fob.engine.parameter_groups import GroupedModel
from pytorch_fob.optimizers.adamcpr_fast.adam_cpr_fast import AdamCPRfast
from pytorch_fob.optimizers.lr_schedulers import get_lr_scheduler, warmup_split_from_config




def configure_optimizers(model: GroupedModel, config: OptimizerConfig) -> OptimizerLRScheduler:
    lr = config.learning_rate
    weight_decay = 1.0  # only 1 and 0 are valid

    parameter_groups = model.grouped_parameters(lr=lr, weight_decay=weight_decay)

    lr_warmup_steps, _ = warmup_split_from_config(config)
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
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        kappa_init_param=kappa_init_param,
        kappa_init_method=kappa_init_method,
        reg_function=config.reg_function,
        kappa_update=config.kappa_update
    )
    scheduler = get_lr_scheduler(optimizer, config)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": config.lr_scheduler.interval
        }
    }
