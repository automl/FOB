from lightning import Callback
from runtime import DatasetArgs
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.cora import data
from workloads.cora import model


def get_datamodule(dataset_args: DatasetArgs) -> WorkloadDataModule:
    return data.CoraDataModule(dataset_args)


def get_workload(submission: Submission, dataset_args: DatasetArgs) -> tuple[WorkloadModel, WorkloadDataModule]:
    datamodule = get_datamodule(dataset_args)
    batch_size = datamodule.batch_size
    return model.CoraModel(submission, batch_size), datamodule


def get_callbacks() -> list[Callback]:
    return []
