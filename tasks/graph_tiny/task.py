from engine.configs import TaskConfig
from optimizers import Optimizer
from tasks import TaskModel, TaskDataModule
from tasks.graph_tiny import data
from tasks.graph_tiny import model


def get_datamodule(config: TaskConfig) -> TaskDataModule:
    return data.CoraDataModule(config)


def get_task(optimizer: Optimizer, config: TaskConfig) -> tuple[TaskModel, TaskDataModule]:
    datamodule = get_datamodule(config)
    return model.CoraModel(optimizer, config), datamodule
