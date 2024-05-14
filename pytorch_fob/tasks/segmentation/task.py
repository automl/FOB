from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.optimizers import Optimizer
from pytorch_fob.tasks import TaskModel, TaskDataModule
from pytorch_fob.tasks.segmentation.data import SegmentationDataModule
from pytorch_fob.tasks.segmentation.model import SegmentationModel


def get_datamodule(config: TaskConfig) -> TaskDataModule:
    return SegmentationDataModule(config)


def get_task(optimizer: Optimizer, config: TaskConfig) -> tuple[TaskModel, TaskDataModule]:
    dm = SegmentationDataModule(config)
    return SegmentationModel(optimizer, dm.id2label, dm.label2id, config), dm
