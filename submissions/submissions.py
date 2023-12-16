import json
import importlib
from pathlib import Path
from typing import Any
from torch.nn import Module
from lightning.pytorch.utilities.types import OptimizerLRScheduler

def import_submission(name: str):
    return importlib.import_module(f"submissions.{name}.submission")


def submission_names() -> list[str]:
    EXCLUDE = ["__pycache__", "template"]
    return [d.name for d in Path(__file__).parent.iterdir() if d.is_dir() and d.name not in EXCLUDE]


class Submission():
    def __init__(self, hyperparameter_path: Path) -> None:
        self.hyperparameter_path = hyperparameter_path

    def configure_optimizers(self, model: Module, workload_specs: dict[str, Any]) -> OptimizerLRScheduler:
        raise NotImplementedError("Each submission must define this method themselves.")

    def get_hyperparameters(self) -> dict[str, Any]:
        # TODO: random sampling from search space
        with open(self.hyperparameter_path, "r", encoding="utf8") as fp:
            hyperarameters = json.load(fp)
        return hyperarameters
