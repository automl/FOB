from lightning.pytorch.utilities.types import OptimizerLRScheduler
from schedulefree import AdamWScheduleFreeClosure

from pytorch_fob.engine.configs import OptimizerConfig
from pytorch_fob.engine.parameter_groups import GroupedModel
from pytorch_fob.optimizers.lr_schedulers import warmup_split_from_config


def configure_optimizers(model: GroupedModel, config: OptimizerConfig) -> OptimizerLRScheduler:
    lr = config.learning_rate
    weight_decay = config.weight_decay
    parameter_groups = model.grouped_parameters(lr=lr, weight_decay=weight_decay)

    lr_warmup_steps, _ = warmup_split_from_config(config)
    optimizer = AdamWScheduleFreeClosure(
        parameter_groups,
        lr=lr,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=weight_decay,
        warmup_steps=lr_warmup_steps,
        r=config.r,
        weight_lr_power=config.weight_lr_power,
    )
    return optimizer
