from engine.configs import TaskConfig
from optimizers import Optimizer
from tasks import TaskModel, TaskDataModule
from tasks.tabular.data import TabularDataModule
from tasks.tabular.model import TabularModel


def get_datamodule(config: TaskConfig) -> TaskDataModule:
    return TabularDataModule(config)


def get_task(optimizer: Optimizer, config: TaskConfig) -> tuple[TaskModel, TaskDataModule]:
    return TabularModel(optimizer, config), TabularDataModule(config)
