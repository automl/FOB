from typing import Optional
from pathlib import Path
from torch.nn import Module
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from workloads.specs import SubmissionSpecs
from submissions import Submission
from runtime import RuntimeArgs


def get_submission(runtime_args: RuntimeArgs) -> Submission:
    return SGDBaseline(runtime_args.hyperparameter_path)


class SGDBaseline(Submission):
    def __init__(self, hyperparameter_path: Optional[Path] = None) -> None:
        if hyperparameter_path is None:
            hyperparameter_path = Path(__file__).parent.joinpath("hyperparameters.json")
        super().__init__(hyperparameter_path)

    def configure_optimizers(self, model: Module, workload_specs: SubmissionSpecs) -> OptimizerLRScheduler:
        hparams = self.get_hyperparameters()
        optimizer = SGD(
            params=model.parameters(),
            lr=hparams["learning_rate"],
            momentum=hparams["momentum"],
            weight_decay=hparams["weight_decay"],
            nesterov=hparams["nesterov"]
        )
        step_hint = workload_specs.max_steps if workload_specs.max_steps else workload_specs.max_epochs
        interval = "step" if workload_specs.max_steps else "epoch"
        scheduler = CosineAnnealingLR(optimizer, T_max=step_hint)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval
            }
        }
