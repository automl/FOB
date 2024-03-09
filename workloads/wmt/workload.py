from engine.configs import WorkloadConfig
from optimizers import Optimizer
from workloads import WorkloadModel, WorkloadDataModule
from workloads.wmt.data import WMTDataModule
from workloads.wmt.model import WMTModel


def get_datamodule(workload_config: WorkloadConfig) -> WorkloadDataModule:
    return WMTDataModule(workload_config)


def get_workload(optimizer: Optimizer, workload_config: WorkloadConfig) -> tuple[WorkloadModel, WorkloadDataModule]:
    data_module = WMTDataModule(workload_config)
    return WMTModel(optimizer, data_module, workload_config), data_module
