from lightning import Callback
from bob.runtime import DatasetArgs
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.coco.data import COCODataModule
from workloads.coco.model import COCODetectionModel


def get_datamodule(dataset_args: DatasetArgs) -> WorkloadDataModule:
    return COCODataModule(dataset_args)


def get_workload(submission: Submission, dataset_args: DatasetArgs) -> tuple[WorkloadModel, WorkloadDataModule]:
    # TODO: pass gts for eval
    return COCODetectionModel(submission), get_datamodule(dataset_args)

def get_callbacks() -> list[Callback]:
    return []
