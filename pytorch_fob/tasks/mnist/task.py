from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.optimizers import Optimizer
from pytorch_fob.tasks import TaskDataModule, TaskModel
from pytorch_fob.tasks.mnist import data
from pytorch_fob.tasks.mnist import model


def get_datamodule(config: TaskConfig) -> TaskDataModule:
    return data.MNISTDataModule(config)

def get_task(optimizer: Optimizer, config: TaskConfig) -> tuple[TaskModel, TaskDataModule]:
    return model.MNISTModel(optimizer, config), get_datamodule(config)
