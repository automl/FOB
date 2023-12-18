from typing import Any
from pathlib import Path
from lightning import LightningDataModule
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.coco.data import COCODataModule
from workloads.coco.model import COCODetectionModel


def get_datamodule(data_dir: Path) -> LightningDataModule:
    return COCODataModule(data_dir)


def get_model(submission: Submission) -> WorkloadModel:
    return COCODetectionModel(submission)


def get_specs(workload: WorkloadModel, datamodule: WorkloadDataModule) -> dict[str, Any]:
    return dict(workload.get_specs(), **(datamodule.get_specs()))
