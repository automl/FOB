from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from engine.parameter_groups import GroupedModel
from engine.configs import OptimizerConfig


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
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.max_steps,
        eta_min=config.eta_min_factor * lr
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step"
        }
    }
