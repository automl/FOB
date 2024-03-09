from engine.configs import TaskConfig
from optimizers import Optimizer
from tasks import TaskDataModule, TaskModel
from tasks.template import data
from tasks.template import model


def get_datamodule(config: TaskConfig) -> TaskDataModule:
    return data.TemplateDataModule(config)

def get_task(optimizer: Optimizer, config: TaskConfig) -> tuple[TaskModel, TaskDataModule]:
    return model.TemplateModel(optimizer, config), get_datamodule(config)
