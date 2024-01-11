from lightning import Callback
from runtime import DatasetArgs
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.coco.data import COCODataModule
from workloads.coco.model import COCODetectionModel
from workloads.coco.callbacks import COCOEvalSummarize


def get_datamodule(dataset_args: DatasetArgs) -> WorkloadDataModule:
    return COCODataModule(dataset_args)


def get_workload(submission: Submission, dataset_args: DatasetArgs) -> tuple[WorkloadModel, WorkloadDataModule]:
    dm = COCODataModule(dataset_args)
    return COCODetectionModel(submission, dm.eval_gt_data()), dm

def get_callbacks() -> list[Callback]:
    return [COCOEvalSummarize()]
