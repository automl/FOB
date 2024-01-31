from typing import Any
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from submissions import Submission
from runtime.parameter_groups import GroupedModel
from runtime.specs import SubmissionSpecs
from runtime import RuntimeArgs


def get_submission(runtime_args: RuntimeArgs) -> Submission:
    return SGDBaseline(runtime_args.hyperparameter_path)


class SGDBaseline(Submission):

    def configure_optimizers(self, model: GroupedModel, workload_specs: SubmissionSpecs) -> OptimizerLRScheduler:
        hparams = self.get_hyperparameters()
        lr=hparams["learning_rate"]
        weight_decay=hparams["weight_decay"]
        optimizer = SGD(
            params=model.grouped_parameters(lr=lr, weight_decay=weight_decay),
            lr=lr,
            momentum=hparams["momentum"],
            weight_decay=weight_decay,
            nesterov=hparams["nesterov"]
        )
        step_hint = workload_specs.max_steps if workload_specs.max_steps else workload_specs.max_epochs
        interval = "step" if workload_specs.max_steps else "epoch"
        scheduler = CosineAnnealingLR(optimizer, T_max=step_hint, eta_min=hparams["eta_min_factor"]*lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval
            }
        }
