from engine.configs import WorkloadConfig
from optimizers import Optimizer
from workloads import WorkloadModel, WorkloadDataModule
from workloads.tabular.data import TabularDataModule
from workloads.tabular.model import TabularModel


def get_datamodule(workload_config: WorkloadConfig) -> WorkloadDataModule:
    return TabularDataModule(workload_config)


def get_workload(optimizer: Optimizer, workload_config: WorkloadConfig) -> tuple[WorkloadModel, WorkloadDataModule]:
    return TabularModel(optimizer, workload_config), TabularDataModule(workload_config)
