from lightning import Callback
from runtime import DatasetArgs
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.imagenet64.data import ImagenetDataModule
from workloads.imagenet64.model import ImagenetModel


def get_datamodule(dataset_args: DatasetArgs) -> WorkloadDataModule:
    return ImagenetDataModule(dataset_args)


def get_workload(submission: Submission, dataset_args: DatasetArgs) -> tuple[WorkloadModel, WorkloadDataModule]:
    return ImagenetModel(submission), get_datamodule(dataset_args)


def get_callbacks() -> list[Callback]:
    return []
