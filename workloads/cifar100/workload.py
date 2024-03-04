from lightning import Callback
from runtime.configs import WorkloadConfig
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.cifar100 import data
from workloads.cifar100 import model


def get_datamodule(workload_config: WorkloadConfig) -> WorkloadDataModule:
    return data.CIFAR100DataModule(workload_config)


def get_workload(submission: Submission, workload_config: WorkloadConfig) -> tuple[WorkloadModel, WorkloadDataModule]:
    return model.CIFAR100Model(submission, workload_config), get_datamodule(workload_config)


def get_callbacks() -> list[Callback]:
    return []
