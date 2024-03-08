from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import SGD
from runtime.parameter_groups import GroupedModel
from runtime.configs import SubmissionConfig


def configure_optimizers(model: GroupedModel, config: SubmissionConfig) -> OptimizerLRScheduler:
    lr = config.learning_rate
    return {"optimizer": SGD(model.grouped_parameters(lr=lr), lr=lr)}
