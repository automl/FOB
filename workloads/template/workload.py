from bob.runtime import RuntimeArgs
from workloads import WorkloadDataModule, WorkloadModel
from submissions import Submission

import workloads.template.data as data
import workloads.template.model as model


def get_datamodule(runtime_args: RuntimeArgs) -> WorkloadDataModule:
    return data.TemplateDataModule(runtime_args)


def get_model(submission: Submission) -> WorkloadModel:
    return model.TemplateModel(submission)
