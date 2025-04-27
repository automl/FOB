from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.optimizers import Optimizer
from pytorch_fob.tasks import TaskDataModule, TaskModel
from pytorch_fob.tasks.cifar5m.data import CIFAR5MDataModule
from pytorch_fob.tasks.cifar5m.model import CIFAR5MModel


def get_datamodule(config: TaskConfig) -> TaskDataModule:
    return CIFAR5MDataModule(config)


def get_task(optimizer: Optimizer, config: TaskConfig) -> tuple[TaskModel, TaskDataModule]:
    return CIFAR5MModel(optimizer, config), get_datamodule(config)
