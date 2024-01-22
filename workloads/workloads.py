import importlib
from typing import Any
from pathlib import Path
from lightning import LightningModule, LightningDataModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.utils.data import DataLoader
from submissions import Submission
from runtime.parameter_groups import GroupedModel
from runtime.specs import RuntimeSpecs, to_submission_specs
from runtime import DatasetArgs


def import_workload(name: str):
    return importlib.import_module(f"workloads.{name}.workload")


def workload_names() -> list[str]:
    EXCLUDE = ["__pycache__"]
    return [d.name for d in Path(__file__).parent.iterdir() if d.is_dir() and d.name not in EXCLUDE]


class WorkloadModel(LightningModule):
    def __init__(self, model: nn.Module | GroupedModel, submission: Submission, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.submission = submission
        self.model = model if isinstance(model, GroupedModel) else GroupedModel(model)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        specs = to_submission_specs(self.get_specs())
        return self.submission.configure_optimizers(self.model, specs)

    def get_specs(self) -> RuntimeSpecs:
        raise NotImplementedError("Each workload has its own specs.")


class WorkloadDataModule(LightningDataModule):
    def __init__(self, dataset_args: DatasetArgs) -> None:
        super().__init__()
        self.workers = min(dataset_args.workers, 16)
        self.data_dir = dataset_args.data_dir
        self.data_train: Any
        self.data_val: Any
        self.data_test: Any
        self.data_predict: Any
        self.batch_size: int
        self.collate_fn = None

    def check_dataset(self, data):
        """Make sure that all workloads have correctly configured their data sets"""
        if not data:
            raise NotImplementedError("Each workload has its own data set")
        if not self.batch_size or self.batch_size < 1:
            raise NotImplementedError("Each workload configures its own batch_size. \
                                      Please set it explicitely, to avoid confusion.")

    def train_dataloader(self):
        self.check_dataset(self.data_train)
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        self.check_dataset(self.data_val)
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        self.check_dataset(self.data_test)
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn
        )

    def predict_dataloader(self):
        self.check_dataset(self.data_predict)
        return DataLoader(
            self.data_predict,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn
        )
