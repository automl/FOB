from lightning import Callback
from runtime import DatasetArgs
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.cora import data
from workloads.cora import model


def get_datamodule(dataset_args: DatasetArgs) -> WorkloadDataModule:
    return data.CoraDataModule(dataset_args)


def get_workload(submission: Submission, dataset_args: DatasetArgs) -> tuple[WorkloadModel, WorkloadDataModule]:
    return model.CoraModel(submission), get_datamodule(dataset_args)

def get_callbacks() -> list[Callback]:
    return []
