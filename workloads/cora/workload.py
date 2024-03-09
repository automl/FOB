from engine.configs import WorkloadConfig
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.cora import data
from workloads.cora import model


def get_datamodule(workload_config: WorkloadConfig) -> WorkloadDataModule:
    return data.CoraDataModule(workload_config)


def get_workload(submission: Submission, workload_config: WorkloadConfig) -> tuple[WorkloadModel, WorkloadDataModule]:
    datamodule = get_datamodule(workload_config)
    return model.CoraModel(submission, workload_config), datamodule
