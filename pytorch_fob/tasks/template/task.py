from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.optimizers import Optimizer
from pytorch_fob.tasks import TaskDataModule, TaskModel
from pytorch_fob.tasks.template import data
from pytorch_fob.tasks.template import model


def get_datamodule(config: TaskConfig) -> TaskDataModule:
    return data.TemplateDataModule(config)


def get_task(optimizer: Optimizer, config: TaskConfig) -> tuple[TaskModel, TaskDataModule]:
    return model.TemplateModel(optimizer, config), get_datamodule(config)
