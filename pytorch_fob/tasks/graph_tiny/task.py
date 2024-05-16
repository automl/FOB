from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.optimizers import Optimizer
from pytorch_fob.tasks import TaskModel, TaskDataModule
from pytorch_fob.tasks.graph_tiny import data
from pytorch_fob.tasks.graph_tiny import model


def get_datamodule(config: TaskConfig) -> TaskDataModule:
    return data.CoraDataModule(config)


def get_task(optimizer: Optimizer, config: TaskConfig) -> tuple[TaskModel, TaskDataModule]:
    datamodule = get_datamodule(config)
    return model.CoraModel(optimizer, config), datamodule
