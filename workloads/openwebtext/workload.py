from lightning import Callback
from runtime import DatasetArgs
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.openwebtext import data
from workloads.openwebtext import model


def get_datamodule(dataset_args: DatasetArgs) -> WorkloadDataModule:
    return data.OpenWebTextDataModule(dataset_args)


def get_workload(submission: Submission, dataset_args: DatasetArgs) -> tuple[WorkloadModel, WorkloadDataModule]:
    return model.OpenWebTextModel(submission), get_datamodule(dataset_args)

def get_callbacks() -> list[Callback]:
    return []
