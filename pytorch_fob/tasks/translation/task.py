from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.optimizers import Optimizer
from pytorch_fob.tasks import TaskModel, TaskDataModule
from pytorch_fob.tasks.translation.data import WMTDataModule
from pytorch_fob.tasks.translation.model import WMTModel


def get_datamodule(config: TaskConfig) -> TaskDataModule:
    return WMTDataModule(config)


def get_task(optimizer: Optimizer, config: TaskConfig) -> tuple[TaskModel, TaskDataModule]:
    data_module = WMTDataModule(config)
    return WMTModel(optimizer, data_module, config), data_module
