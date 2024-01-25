from lightning import Callback
from runtime import DatasetArgs
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.shakespeare import data
from workloads.shakespeare import model


def get_datamodule(dataset_args: DatasetArgs) -> WorkloadDataModule:
    return data.ShakespeareDataModule(dataset_args)


def get_workload(submission: Submission, dataset_args: DatasetArgs) -> tuple[WorkloadModel, WorkloadDataModule]:
    return model.ShakespeareModel(submission), get_datamodule(dataset_args)


def get_callbacks() -> list[Callback]:
    return []
