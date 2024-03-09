from engine.configs import WorkloadConfig
from optimizers import Optimizer
from workloads import WorkloadModel, WorkloadDataModule
from workloads.segmentation.data import SegmentationDataModule
from workloads.segmentation.model import SegmentationModel


def get_datamodule(workload_config: WorkloadConfig) -> WorkloadDataModule:
    return SegmentationDataModule(workload_config)


def get_workload(optimizer: Optimizer, workload_config: WorkloadConfig) -> tuple[WorkloadModel, WorkloadDataModule]:
    dm = SegmentationDataModule(workload_config)
    return SegmentationModel(optimizer, dm.data_dir, dm.id2label, dm.label2id, workload_config), dm
