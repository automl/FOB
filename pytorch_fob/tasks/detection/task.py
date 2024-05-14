from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.optimizers import Optimizer
from pytorch_fob.tasks import TaskModel, TaskDataModule
from pytorch_fob.tasks.detection.data import COCODataModule
from pytorch_fob.tasks.detection.model import COCODetectionModel


def get_datamodule(config: TaskConfig) -> TaskDataModule:
    return COCODataModule(config)


def get_task(optimizer: Optimizer, config: TaskConfig) -> tuple[TaskModel, TaskDataModule]:
    dm = COCODataModule(config)
    return COCODetectionModel(optimizer, config, dm.eval_gt_data()), dm
