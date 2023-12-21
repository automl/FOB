from typing import Any
from pathlib import Path
from lightning import LightningDataModule
from workloads import WorkloadModel, WorkloadDataModule
from submissions import Submission
from bob.runtime import DatasetArgs

import workloads.openwebtext.data as data
import workloads.openwebtext.model as model


def get_datamodule(dataset_args: DatasetArgs) -> WorkloadDataModule:
    return data.OpenWebTextDataModule(dataset_args)


def get_model(submission: Submission) -> WorkloadModel:
    return model.OpenWebTextModel(submission)
