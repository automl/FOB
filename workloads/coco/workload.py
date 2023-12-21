from typing import Any
from pathlib import Path
from lightning import LightningDataModule
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.coco.data import COCODataModule
from workloads.coco.model import COCODetectionModel
from bob.runtime import DatasetArgs


def get_datamodule(dataset_args: DatasetArgs) -> WorkloadDataModule:
    return COCODataModule(dataset_args)


def get_model(submission: Submission) -> WorkloadModel:
    return COCODetectionModel(submission)
