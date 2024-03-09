from engine.configs import WorkloadConfig
from optimizers import Optimizer
from workloads import WorkloadModel, WorkloadDataModule
from workloads.cifar100 import data
from workloads.cifar100 import model


def get_datamodule(workload_config: WorkloadConfig) -> WorkloadDataModule:
    return data.CIFAR100DataModule(workload_config)


def get_workload(optimizer: Optimizer, workload_config: WorkloadConfig) -> tuple[WorkloadModel, WorkloadDataModule]:
    return model.CIFAR100Model(optimizer, workload_config), get_datamodule(workload_config)
