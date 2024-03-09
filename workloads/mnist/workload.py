from engine.configs import WorkloadConfig
from submissions import Submission
from workloads import WorkloadDataModule, WorkloadModel
from workloads.mnist import data
from workloads.mnist import model


def get_datamodule(workload_config: WorkloadConfig) -> WorkloadDataModule:
    return data.MNISTDataModule(workload_config)

def get_workload(submission: Submission, workload_config: WorkloadConfig) -> tuple[WorkloadModel, WorkloadDataModule]:
    return model.MNISTModel(submission, workload_config), get_datamodule(workload_config)
