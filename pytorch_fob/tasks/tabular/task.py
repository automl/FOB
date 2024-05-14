from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.optimizers import Optimizer
from pytorch_fob.tasks import TaskModel, TaskDataModule
from pytorch_fob.tasks.tabular.data import TabularDataModule
from pytorch_fob.tasks.tabular.model import TabularModel


def get_datamodule(config: TaskConfig) -> TaskDataModule:
    return TabularDataModule(config)


def get_task(optimizer: Optimizer, config: TaskConfig) -> tuple[TaskModel, TaskDataModule]:
    return TabularModel(optimizer, config), TabularDataModule(config)
