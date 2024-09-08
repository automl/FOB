from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR
from pytorch_fob.engine.parameter_groups import GroupedModel
from pytorch_fob.engine.configs import OptimizerConfig


def configure_optimizers(model: GroupedModel, config: OptimizerConfig) -> OptimizerLRScheduler:
    lr = config.learning_rate
    optimizer = SGD(model.grouped_parameters(lr=lr), lr=lr)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": LinearLR(optimizer, total_iters=config.max_steps),
            "interval": config.lr_scheduler.interval
        }
    }
