from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lion_pytorch import Lion
from pytorch_fob.engine.configs import OptimizerConfig
from pytorch_fob.engine.parameter_groups import GroupedModel
from pytorch_fob.optimizers.lr_schedulers import get_lr_scheduler


def configure_optimizers(model: GroupedModel, config: OptimizerConfig) -> OptimizerLRScheduler:
    lr = config.learning_rate
    weight_decay = config.weight_decay
    parameter_groups = model.grouped_parameters(lr=lr, weight_decay=weight_decay)
    optimizer = Lion(
        params=parameter_groups,
        lr=lr,
        betas=(config.beta1, config.beta2),
        weight_decay=weight_decay,
        decoupled_weight_decay=config.decoupled_weight_decay
    )
    lr_scheduler = get_lr_scheduler(optimizer, config)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": lr_scheduler,
            "interval": config.lr_scheduler.interval
        }
    }