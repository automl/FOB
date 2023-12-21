import importlib
from typing import Any
from pathlib import Path
from lightning import LightningModule, LightningDataModule
from torch.utils.data import DataLoader
import torch.nn as nn
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from submissions import Submission
from bob.runtime import DatasetArgs


def import_workload(name: str):
    return importlib.import_module(f"workloads.{name}.workload")


def workload_names() -> list[str]:
    EXCLUDE = ["__pycache__"]
    return [d.name for d in Path(__file__).parent.iterdir() if d.is_dir() and d.name not in EXCLUDE]


class WorkloadModel(LightningModule):
    def __init__(self, model: nn.Module, submission: Submission, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.submission = submission
        self.model = model

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.submission.configure_optimizers(self.model, self.get_specs())

    def get_specs(self) -> dict[str, Any]:
        raise NotImplementedError("Each workload has its own specs.")


class WorkloadDataModule(LightningDataModule):
    def __init__(self, dataset_args: DatasetArgs) -> None:
        super().__init__()
        self.workers = dataset_args.cpu_cores - 1
        self.data_dir = dataset_args.data_dir
        self.data_train: Any
        self.data_val: Any
        self.data_test: Any
        self.data_predict: Any
        self.batch_size: int

    def check_dataset(self, data):
        """Make sure that all workloads have correctly configured their data sets"""
        if not data:
            raise NotImplementedError("Each workload has its own data set")
        if not self.batch_size or self.batch_size < 1:
            raise NotImplementedError("Each workload configures its own batch_size. Please set it explicitely, to avoid confusion.")


    def train_dataloader(self):
        self.check_dataset(self.data_train)
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.workers)

    def val_dataloader(self):
        self.check_dataset(self.data_val)
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.workers)

    def test_dataloader(self):
        self.check_dataset(self.data_test)
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.workers)

    def predict_dataloader(self):
        self.check_dataset(self.data_predict)
        return DataLoader(self.data_predict, batch_size=self.batch_size)

    def get_specs(self) -> dict[str, Any]:
        raise NotImplementedError("Each workload has its own specs.")


def combine_specs(workload: WorkloadModel, datamodule: WorkloadDataModule) -> dict[str, Any]:
    return dict(workload.get_specs(), **(datamodule.get_specs()))