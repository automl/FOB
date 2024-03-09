from engine.configs import WorkloadConfig
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.coco.data import COCODataModule
from workloads.coco.model import COCODetectionModel


def get_datamodule(workload_config: WorkloadConfig) -> WorkloadDataModule:
    return COCODataModule(workload_config)


def get_workload(submission: Submission, workload_config: WorkloadConfig) -> tuple[WorkloadModel, WorkloadDataModule]:
    dm = COCODataModule(workload_config)
    return COCODetectionModel(submission, workload_config, dm.eval_gt_data()), dm
