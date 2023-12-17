from typing import Any
from pathlib import Path
from lightning import LightningDataModule
from workloads import WorkloadModel, WorkloadDataModule
from submissions import Submission

import workloads.openwebtext.data as data
import workloads.openwebtext.model as model


def get_datamodule(data_dir: Path) -> LightningDataModule:
    return data.OpenWebTextDataModule(data_dir)


def get_model(submission: Submission) -> WorkloadModel:
    return model.OpenWebTextModel(submission)


def get_specs(workload: WorkloadModel, datamodule: WorkloadDataModule) -> dict[str, Any]:
    return dict(workload.get_specs(), **(datamodule.get_specs()))