from lightning import Callback
from runtime.configs import WorkloadConfig
from submissions import Submission
from workloads import WorkloadDataModule, WorkloadModel
from workloads.template import data
from workloads.template import model


def get_datamodule(workload_config: WorkloadConfig) -> WorkloadDataModule:
    return data.TemplateDataModule(workload_config)

def get_workload(submission: Submission, workload_config: WorkloadConfig) -> tuple[WorkloadModel, WorkloadDataModule]:
    return model.TemplateModel(submission, workload_config), get_datamodule(workload_config)

def get_callbacks() -> list[Callback]:
    return []
