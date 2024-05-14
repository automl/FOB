from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.optimizers import Optimizer
from pytorch_fob.tasks import TaskModel, TaskDataModule
from pytorch_fob.tasks.classification_small.data import CIFAR100DataModule
from pytorch_fob.tasks.classification_small.model import CIFAR100Model


def get_datamodule(config: TaskConfig) -> TaskDataModule:
    return CIFAR100DataModule(config)


def get_task(optimizer: Optimizer, config: TaskConfig) -> tuple[TaskModel, TaskDataModule]:
    return CIFAR100Model(optimizer, config), get_datamodule(config)
