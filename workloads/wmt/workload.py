from runtime.configs import WorkloadConfig
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.wmt.data import WMTDataModule
from workloads.wmt.model import WMTModel


def get_datamodule(workload_config: WorkloadConfig) -> WorkloadDataModule:
    return WMTDataModule(workload_config)


def get_workload(submission: Submission, workload_config: WorkloadConfig) -> tuple[WorkloadModel, WorkloadDataModule]:
    data_module = WMTDataModule(workload_config)
    return WMTModel(submission, data_module, workload_config), data_module
