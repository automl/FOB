from bob.runtime import DatasetArgs
from submissions import Submission
from workloads import WorkloadDataModule, WorkloadModel
from workloads.template import data
from workloads.template import model


def get_datamodule(dataset_args: DatasetArgs) -> WorkloadDataModule:
    return data.TemplateDataModule(dataset_args)


def get_workload(submission: Submission, dataset_args: DatasetArgs) -> tuple[WorkloadModel, WorkloadDataModule]:
    return model.TemplateModel(submission), get_datamodule(dataset_args)
