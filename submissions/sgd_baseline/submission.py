from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from runtime.parameter_groups import GroupedModel
from runtime.configs import SubmissionConfig, WorkloadConfig


def configure_optimizers(
        model: GroupedModel,
        workload_config: WorkloadConfig,
        submission_config: SubmissionConfig
    ) -> OptimizerLRScheduler:
    hparams = submission_config
    lr=hparams["learning_rate"]
    weight_decay=hparams["weight_decay"]
    optimizer = SGD(
        params=model.grouped_parameters(lr=lr, weight_decay=weight_decay),
        lr=lr,
        momentum=hparams["momentum"],
        weight_decay=weight_decay,
        nesterov=hparams["nesterov"]
    )
    step_hint = workload_config.max_steps if workload_config.max_steps else workload_config.max_epochs
    interval = "step" if workload_config.max_steps else "epoch"
    scheduler = CosineAnnealingLR(optimizer, T_max=step_hint, eta_min=hparams["eta_min_factor"]*lr)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": interval
        }
    }
