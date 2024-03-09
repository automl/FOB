from engine.configs import TaskConfig
from optimizers import Optimizer
from tasks import TaskModel, TaskDataModule
from tasks.coco.data import COCODataModule
from tasks.coco.model import COCODetectionModel


def get_datamodule(config: TaskConfig) -> TaskDataModule:
    return COCODataModule(config)


def get_task(optimizer: Optimizer, config: TaskConfig) -> tuple[TaskModel, TaskDataModule]:
    dm = COCODataModule(config)
    return COCODetectionModel(optimizer, config, dm.eval_gt_data()), dm
