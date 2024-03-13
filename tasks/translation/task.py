from engine.configs import TaskConfig
from optimizers import Optimizer
from tasks import TaskModel, TaskDataModule
from tasks.translation.data import WMTDataModule
from tasks.translation.model import WMTModel


def get_datamodule(config: TaskConfig) -> TaskDataModule:
    return WMTDataModule(config)


def get_task(optimizer: Optimizer, config: TaskConfig) -> tuple[TaskModel, TaskDataModule]:
    data_module = WMTDataModule(config)
    return WMTModel(optimizer, data_module, config), data_module
