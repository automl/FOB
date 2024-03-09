from engine.configs import WorkloadConfig
from optimizers import Optimizer
from workloads import WorkloadModel, WorkloadDataModule
from workloads.cora import data
from workloads.cora import model


def get_datamodule(workload_config: WorkloadConfig) -> WorkloadDataModule:
    return data.CoraDataModule(workload_config)


def get_workload(optimizer: Optimizer, workload_config: WorkloadConfig) -> tuple[WorkloadModel, WorkloadDataModule]:
    datamodule = get_datamodule(workload_config)
    return model.CoraModel(optimizer, workload_config), datamodule
