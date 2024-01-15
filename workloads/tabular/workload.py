from lightning import Callback
from runtime import DatasetArgs
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.tabular.data import TabularDataModule
from workloads.tabular.model import TabularModel


def get_datamodule(dataset_args: DatasetArgs) -> WorkloadDataModule:
    return TabularDataModule(dataset_args)


def get_workload(submission: Submission, dataset_args: DatasetArgs) -> tuple[WorkloadModel, WorkloadDataModule]:
    return TabularModel(submission), TabularDataModule(dataset_args)

def get_callbacks() -> list[Callback]:
    return []
