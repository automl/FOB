from typing import Any
from pathlib import Path
from lightning import LightningDataModule
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.coco.data import COCODataModule
from workloads.coco.model import COCODetectionModel
from bob.runtime import RuntimeArgs


def get_datamodule(runtime_args: RuntimeArgs) -> WorkloadDataModule:
    return COCODataModule(runtime_args)


def get_model(submission: Submission) -> WorkloadModel:
    return COCODetectionModel(submission)
