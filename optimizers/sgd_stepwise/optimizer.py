from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from engine.parameter_groups import GroupedModel
from engine.configs import OptimizerConfig

def configure_optimizers(model: GroupedModel, config: OptimizerConfig) -> OptimizerLRScheduler:
    lr = config.learning_rate
    optimizer = SGD(
        model.grouped_parameters(lr=lr, weight_decay=config.weight_decay),
        lr=lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )
    scheduler = StepLR(
        optimizer=optimizer,
        step_size=config.step_size,
        gamma=config.gamma,
        last_epoch=config.last_epoch
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": config.lr_interval
        }
    }
