from bob.runtime import DatasetArgs
from workloads import WorkloadDataModule, WorkloadModel
from submissions import Submission

import workloads.template.data as data
import workloads.template.model as model


def get_datamodule(dataset_args: DatasetArgs) -> WorkloadDataModule:
    return data.TemplateDataModule(dataset_args)


def get_model(submission: Submission) -> WorkloadModel:
    return model.TemplateModel(submission)
