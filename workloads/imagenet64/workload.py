from engine.configs import WorkloadConfig
from optimizers import Optimizer
from workloads import WorkloadModel, WorkloadDataModule
from workloads.imagenet64.data import ImagenetDataModule
from workloads.imagenet64.model import ImagenetModel


def get_datamodule(workload_config: WorkloadConfig) -> WorkloadDataModule:
    return ImagenetDataModule(workload_config)


def get_workload(optimizer: Optimizer, workload_config: WorkloadConfig) -> tuple[WorkloadModel, WorkloadDataModule]:
    return ImagenetModel(optimizer, workload_config), get_datamodule(workload_config)
