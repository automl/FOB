from typing import Any
from pathlib import Path
from lightning import LightningDataModule
from workloads import WorkloadModel, WorkloadDataModule
from submissions import Submission
from bob.runtime import RuntimeArgs

import workloads.ogbg.data as data
import workloads.ogbg.model as model


def get_datamodule(runtime_args: RuntimeArgs) -> WorkloadDataModule:
    return data.OGBGDataModule(runtime_args)


def get_model(submission: Submission) -> WorkloadModel:
    return model.OGBGModel(submission)
