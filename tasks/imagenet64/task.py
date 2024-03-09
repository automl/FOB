from engine.configs import TaskConfig
from optimizers import Optimizer
from tasks import TaskModel, TaskDataModule
from tasks.imagenet64.data import ImagenetDataModule
from tasks.imagenet64.model import ImagenetModel


def get_datamodule(config: TaskConfig) -> TaskDataModule:
    return ImagenetDataModule(config)


def get_task(optimizer: Optimizer, config: TaskConfig) -> tuple[TaskModel, TaskDataModule]:
    return ImagenetModel(optimizer, config), get_datamodule(config)
