from lightning import Callback
from bob.runtime import DatasetArgs
from submissions import Submission
from workloads import WorkloadDataModule, WorkloadModel
from workloads.mnist import data
from workloads.mnist import model

def get_datamodule(dataset_args: DatasetArgs) -> WorkloadDataModule:
    return data.MNISTDataModule(dataset_args)

def get_workload(submission: Submission, dataset_args: DatasetArgs) -> tuple[WorkloadModel, WorkloadDataModule]:
    return model.MNISTModel(submission), get_datamodule(dataset_args)

def get_callbacks() -> list[Callback]:
    return []
