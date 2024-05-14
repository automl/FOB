from engine.configs import TaskConfig
from optimizers import Optimizer
from tasks import TaskModel, TaskDataModule
from tasks.graph import data
from tasks.graph import model


def get_datamodule(config: TaskConfig) -> TaskDataModule:
    return data.OGBGDataModule(config)

def get_task(optimizer: Optimizer, config: TaskConfig) -> tuple[TaskModel, TaskDataModule]:
    datamodule = data.OGBGDataModule(config)
    ogbg_model = model.OGBGModel(optimizer, config)
    return ogbg_model, datamodule
