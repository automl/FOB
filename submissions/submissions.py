import importlib
from pathlib import Path
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from engine.parameter_groups import GroupedModel
from engine.configs import SubmissionConfig


def import_submission(name: str):
    return importlib.import_module(f"submissions.{name}.submission")


def submission_path(name: str) -> Path:
    return Path(__file__).resolve().parent / name


def submission_names() -> list[str]:
    EXCLUDE = ["__pycache__"]
    return [d.name for d in Path(__file__).parent.iterdir() if d.is_dir() and d.name not in EXCLUDE]


class Submission():
    def __init__(self, config: SubmissionConfig) -> None:
        self.config = config

    def configure_optimizers(self, model: GroupedModel) -> OptimizerLRScheduler:
        submission_module = import_submission(self.config.name)
        return submission_module.configure_optimizers(model, self.config)
