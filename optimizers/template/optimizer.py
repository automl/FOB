from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import SGD
from engine.parameter_groups import GroupedModel
from engine.configs import OptimizerConfig


def configure_optimizers(model: GroupedModel, config: OptimizerConfig) -> OptimizerLRScheduler:
    lr = config.learning_rate
    return {"optimizer": SGD(model.grouped_parameters(lr=lr), lr=lr)}
