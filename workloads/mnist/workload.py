from typing import Any
from pathlib import Path
from lightning import LightningDataModule, LightningModule
from workloads import WorkloadModel, WorkloadDataModule
from submissions import Submission

import workloads.mnist.data as data
import workloads.mnist.model as model

def get_datamodule(data_dir: Path) -> LightningDataModule:
    return data.MNISTDataModule(data_dir)

def get_model(submission: Submission) -> LightningModule:
    return model.MNISTModel(submission)

def get_specs(workload: WorkloadModel, datamodule: WorkloadDataModule) -> dict[str, Any]:
    return dict(workload.get_specs(), **(datamodule.get_specs()))