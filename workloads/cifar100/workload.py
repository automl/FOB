from lightning import Callback
from runtime import DatasetArgs
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.cifar100 import data
from workloads.cifar100 import model


def get_datamodule(dataset_args: DatasetArgs) -> WorkloadDataModule:
    return data.CIFAR100DataModule(dataset_args)


def get_workload(submission: Submission, dataset_args: DatasetArgs) -> tuple[WorkloadModel, WorkloadDataModule]:
    return model.CIFAR100Model(submission), get_datamodule(dataset_args)

def get_callbacks() -> list[Callback]:
    return []
