from bob.runtime import RuntimeArgs
from workloads import WorkloadDataModule, WorkloadModel
from submissions import Submission

import workloads.mnist.data as data
import workloads.mnist.model as model

def get_datamodule(runtime_args: RuntimeArgs) -> WorkloadDataModule:
    return data.MNISTDataModule(runtime_args)

def get_model(submission: Submission) -> WorkloadModel:
    return model.MNISTModel(submission)
