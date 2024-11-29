import importlib
import time
from typing import Any, Callable, Optional
from pathlib import Path
from lightning import LightningModule, LightningDataModule
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch
from torch import nn
from torch.utils.data import DataLoader
from pytorch_fob.optimizers import Optimizer
from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.engine.parameter_groups import GroupedModel


def import_task(name: str):
    return importlib.import_module(f"pytorch_fob.tasks.{name}.task")


def task_path(name: str) -> Path:
    return Path(__file__).resolve().parent / name


def task_names() -> list[str]:
    EXCLUDE = ["__pycache__"]
    return [d.name for d in Path(__file__).parent.iterdir() if d.is_dir() and d.name not in EXCLUDE]


class TaskModel(LightningModule):
    def __init__(
            self,
            model: nn.Module | GroupedModel,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.model = model if isinstance(model, GroupedModel) else GroupedModel(model)
        self.optimizer_times_ms = []

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.optimizer.configure_optimizers(self.model)

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer | LightningOptimizer,
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        start = time.time_ns()
        optimizer.step(closure=optimizer_closure)  # type: ignore
        end = time.time_ns()
        duration_ms = (end - start) / 1e6
        self.optimizer_times_ms.append(duration_ms)


class TaskDataModule(LightningDataModule):
    def __init__(self, data_dir) -> None:
        super().__init__()
        self.workers: int = 4
        self.data_dir: Path = data_dir
        self.batch_size: int
        self.data_train: Any
        self.data_val: Any
        self.data_test: Any
        self.data_predict: Any
        self.collate_fn = None

    def check_dataset(self, data):
        """Make sure that all tasks have correctly configured their data sets"""
        if not data:
            raise NotImplementedError("Each task has its own data set")
        if not self.batch_size or self.batch_size < 1:
            raise NotImplementedError("Each task configures its own batch_size. \
                                      Please set it explicitely, to avoid confusion.")

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def train_dataloader(self):
        self.check_dataset(self.data_train)
        return DataLoader(
            self.data_train,
            shuffle=True,
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
