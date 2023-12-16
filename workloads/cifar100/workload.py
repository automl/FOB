from typing import Any
from pathlib import Path
from lightning import LightningDataModule
from workloads import WorkloadModel, WorkloadDataModule
from submissions import Submission

import workloads.cifar100.data as data
import workloads.cifar100.model as model


def get_datamodule(data_dir: Path) -> LightningDataModule:
    return data.CIFAR100DataModule(data_dir)


def get_model(submission: Submission) -> WorkloadModel:
    return model.CIFAR100Model(submission)


def get_specs(workload: WorkloadModel, datamodule: WorkloadDataModule) -> dict[str, Any]:
    return dict(workload.get_specs(), **(datamodule.get_specs()))
