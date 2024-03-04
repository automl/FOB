from lightning import Callback
from runtime.configs import WorkloadConfig
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.imagenet64.data import ImagenetDataModule
from workloads.imagenet64.model import ImagenetModel


def get_datamodule(workload_config: WorkloadConfig) -> WorkloadDataModule:
    return ImagenetDataModule(workload_config)


def get_workload(submission: Submission, workload_config: WorkloadConfig) -> tuple[WorkloadModel, WorkloadDataModule]:
    return ImagenetModel(submission), get_datamodule(workload_config)


def get_callbacks() -> list[Callback]:
    return []
