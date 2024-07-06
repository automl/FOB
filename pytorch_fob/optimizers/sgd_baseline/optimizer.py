from torch.optim import SGD
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from pytorch_fob.engine.parameter_groups import GroupedModel
from pytorch_fob.engine.configs import OptimizerConfig
from pytorch_fob.optimizers.lr_schedulers import get_lr_scheduler


def configure_optimizers(model: GroupedModel, config: OptimizerConfig) -> OptimizerLRScheduler:
    lr = config.learning_rate
    weight_decay = config.weight_decay
    optimizer = SGD(
        params=model.grouped_parameters(lr=lr, weight_decay=weight_decay),
        lr=lr,
        momentum=config.momentum,
        weight_decay=weight_decay,
        nesterov=config.nesterov
    )
    scheduler = get_lr_scheduler(optimizer, config)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": config.lr_scheduler.interval
        }
    }
