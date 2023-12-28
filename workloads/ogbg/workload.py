from lightning import Callback
from bob.runtime import DatasetArgs
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.ogbg import data
from workloads.ogbg import model


def get_datamodule(dataset_args: DatasetArgs) -> WorkloadDataModule:
    return data.OGBGDataModule(dataset_args)


def get_workload(submission: Submission, dataset_args: DatasetArgs) -> tuple[WorkloadModel, WorkloadDataModule]:
    return model.OGBGModel(submission), get_datamodule(dataset_args)

def get_callbacks() -> list[Callback]:
    return []
