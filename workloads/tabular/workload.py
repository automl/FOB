from runtime.configs import WorkloadConfig
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.tabular.data import TabularDataModule
from workloads.tabular.model import TabularModel


def get_datamodule(workload_config: WorkloadConfig) -> WorkloadDataModule:
    return TabularDataModule(workload_config)


def get_workload(submission: Submission, workload_config: WorkloadConfig) -> tuple[WorkloadModel, WorkloadDataModule]:
    return TabularModel(submission, workload_config), TabularDataModule(workload_config)
