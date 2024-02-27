import importlib
from pathlib import Path
from typing import Any
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from runtime.parameter_groups import GroupedModel
from runtime.specs import SubmissionSpecs


def import_submission(name: str):
    return importlib.import_module(f"submissions.{name}.submission")


def submission_path(name: str) -> Path:
    return Path(__file__).resolve().parent / name


def submission_names() -> list[str]:
    EXCLUDE = ["__pycache__"]
    return [d.name for d in Path(__file__).parent.iterdir() if d.is_dir() and d.name not in EXCLUDE]


class Submission():
    def __init__(self, hyperparameters: dict[str, Any]) -> None:
        self.hyperparameters = hyperparameters

    def configure_optimizers(self, model: GroupedModel, workload_specs: SubmissionSpecs) -> OptimizerLRScheduler:
        raise NotImplementedError("Each submission must define this method themselves.")

    def get_hyperparameters(self) -> dict[str, Any]:
        return self.hyperparameters
