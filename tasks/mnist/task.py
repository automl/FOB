from engine.configs import TaskConfig
from optimizers import Optimizer
from tasks import TaskDataModule, TaskModel
from tasks.mnist import data
from tasks.mnist import model


def get_datamodule(config: TaskConfig) -> TaskDataModule:
    return data.MNISTDataModule(config)

def get_task(optimizer: Optimizer, config: TaskConfig) -> tuple[TaskModel, TaskDataModule]:
    return model.MNISTModel(optimizer, config), get_datamodule(config)
