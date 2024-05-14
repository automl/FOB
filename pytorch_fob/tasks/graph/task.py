from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.optimizers import Optimizer
from pytorch_fob.tasks import TaskModel, TaskDataModule
from pytorch_fob.tasks.graph import data
from pytorch_fob.tasks.graph import model


def get_datamodule(config: TaskConfig) -> TaskDataModule:
    return data.OGBGDataModule(config)

def get_task(optimizer: Optimizer, config: TaskConfig) -> tuple[TaskModel, TaskDataModule]:
    datamodule = data.OGBGDataModule(config)
    ogbg_model = model.OGBGModel(optimizer, config)
    return ogbg_model, datamodule
