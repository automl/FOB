from engine.configs import TaskConfig
from optimizers import Optimizer
from tasks import TaskModel, TaskDataModule
from tasks.segmentation.data import SegmentationDataModule
from tasks.segmentation.model import SegmentationModel


def get_datamodule(config: TaskConfig) -> TaskDataModule:
    return SegmentationDataModule(config)


def get_task(optimizer: Optimizer, config: TaskConfig) -> tuple[TaskModel, TaskDataModule]:
    dm = SegmentationDataModule(config)
    return SegmentationModel(optimizer, dm.id2label, dm.label2id, config), dm
