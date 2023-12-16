from typing import Any, Optional
from pathlib import Path
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.nn import Module
from torch.optim import SGD
from submissions import Submission


def get_submission(hyperparameter_path: Optional[Path] = None) -> Submission:
    return TemplateSubmission(hyperparameter_path)


class TemplateSubmission(Submission):
    def __init__(self, hyperparameter_path: Optional[Path] = None) -> None:
        if hyperparameter_path is None:
            hyperparameter_path = Path(__file__).parent.joinpath("hyperparameters.json")
        super().__init__(hyperparameter_path)

    def configure_optimizers(self, model: Module, workload_specs: dict[str, Any]) -> OptimizerLRScheduler:
        hparams = self.get_hyperparameters()
        return {"optimizer": SGD(model.parameters(), lr=hparams["learning_rate"])}
