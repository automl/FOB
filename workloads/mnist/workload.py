from bob.runtime import DatasetArgs
from workloads import WorkloadDataModule, WorkloadModel
from submissions import Submission

import workloads.mnist.data as data
import workloads.mnist.model as model

def get_datamodule(dataset_args: DatasetArgs) -> WorkloadDataModule:
    return data.MNISTDataModule(dataset_args)

def get_model(submission: Submission) -> WorkloadModel:
    return model.MNISTModel(submission)
