from engine.configs import WorkloadConfig
from optimizers import Optimizer
from workloads import WorkloadDataModule, WorkloadModel
from workloads.template import data
from workloads.template import model


def get_datamodule(workload_config: WorkloadConfig) -> WorkloadDataModule:
    return data.TemplateDataModule(workload_config)

def get_workload(optimizer: Optimizer, workload_config: WorkloadConfig) -> tuple[WorkloadModel, WorkloadDataModule]:
    return model.TemplateModel(optimizer, workload_config), get_datamodule(workload_config)
