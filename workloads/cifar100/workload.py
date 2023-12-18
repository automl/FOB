from typing import Any
from pathlib import Path
from lightning import LightningDataModule
from workloads import WorkloadModel, WorkloadDataModule
from submissions import Submission
from bob.runtime import RuntimeArgs

import workloads.cifar100.data as data
import workloads.cifar100.model as model


def get_datamodule(runtime_args: RuntimeArgs) -> LightningDataModule:
    return data.CIFAR100DataModule(runtime_args)


def get_model(submission: Submission) -> WorkloadModel:
    return model.CIFAR100Model(submission)
