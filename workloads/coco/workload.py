from lightning import Callback
from runtime.configs import WorkloadConfig
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.coco.data import COCODataModule
from workloads.coco.model import COCODetectionModel
from workloads.coco.callbacks import COCOEvalSummarize


def get_datamodule(workload_config: WorkloadConfig) -> WorkloadDataModule:
    return COCODataModule(workload_config)


def get_workload(submission: Submission, workload_config: WorkloadConfig) -> tuple[WorkloadModel, WorkloadDataModule]:
    dm = COCODataModule(workload_config)
    return COCODetectionModel(submission, dm.eval_gt_data()), dm


def get_callbacks() -> list[Callback]:
    return [COCOEvalSummarize()]
