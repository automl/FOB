from lightning import Callback
from runtime import DatasetArgs
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.wmt.data import WMTDataModule
from workloads.wmt.model import WMTModel


def get_datamodule(dataset_args: DatasetArgs) -> WorkloadDataModule:
    return WMTDataModule(dataset_args)


def get_workload(submission: Submission, dataset_args: DatasetArgs) -> tuple[WorkloadModel, WorkloadDataModule]:
    data_module = WMTDataModule(dataset_args)
    return WMTModel(submission, data_module), data_module

def get_callbacks() -> list[Callback]:
    return []
