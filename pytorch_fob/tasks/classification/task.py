from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.optimizers import Optimizer
from pytorch_fob.tasks import TaskModel, TaskDataModule
from pytorch_fob.tasks.classification.data import ImagenetDataModule
from pytorch_fob.tasks.classification.model import ImagenetModel


def get_datamodule(config: TaskConfig) -> TaskDataModule:
    return ImagenetDataModule(config)


def get_task(optimizer: Optimizer, config: TaskConfig) -> tuple[TaskModel, TaskDataModule]:
    return ImagenetModel(optimizer, config), get_datamodule(config)
