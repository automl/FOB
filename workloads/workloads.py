import importlib
from typing import Any
from pathlib import Path
from lightning import LightningModule, LightningDataModule
from submissions import Submission
from torch.nn import Module
from lightning.pytorch.utilities.types import OptimizerLRScheduler


def import_workload(name: str):
    return importlib.import_module(f"workloads.{name}.workload")


def workload_names() -> list[str]:
    EXCLUDE = ["__pycache__", "template"]
    return [d.name for d in Path(__file__).parent.iterdir() if d.is_dir() and d.name not in EXCLUDE]


class WorkloadModel(LightningModule):
    def __init__(self, model: Module, submission: Submission, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.submission = submission
        self.model = model

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.submission.configure_optimizers(self.model, self.get_specs())

    def get_specs(self) -> dict[str, Any]:
        raise NotImplementedError("Each workload has its own specs.")


class WorkloadDataModule(LightningDataModule):
    def __init__(self) -> None:
        super().__init__()

    def get_specs(self) -> dict[str, Any]:
        raise NotImplementedError("Each workload has its own specs.")
