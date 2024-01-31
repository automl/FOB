import math
from typing import Any
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from submissions import Submission
from runtime import RuntimeArgs
from runtime.specs import SubmissionSpecs
from runtime.parameter_groups import GroupedModel


def get_submission(runtime_args: RuntimeArgs) -> Submission:
    return AdamWBaseline(runtime_args.hyperparameter_path)


def cosine_warmup(
        step_hint: int,
        warmup_factor: float,
        eta_min: float,
        optimizer
    ) -> SequentialLR | CosineAnnealingLR:
    if step_hint < 1:
        raise ValueError("step hint should be at least 1!")
    warmup_steps = math.ceil(warmup_factor * step_hint)
    if warmup_steps == 0:
        print("warmup = 0: using CosineAnnealingLR only")
        return CosineAnnealingLR(optimizer, T_max=step_hint)
    warmup = LinearLR(
        optimizer, start_factor=1e-10, end_factor=1., total_iters=warmup_steps)
    cosine_steps = max(step_hint - warmup_steps, 1)
    cosine_decay = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=eta_min)
    return SequentialLR(
        optimizer, schedulers=[warmup, cosine_decay], milestones=[warmup_steps])


class AdamWBaseline(Submission):

    def configure_optimizers(self, model: GroupedModel, workload_specs: SubmissionSpecs) -> OptimizerLRScheduler:
        hparams = self.get_hyperparameters()
        lr=hparams["learning_rate"]
        weight_decay=hparams["weight_decay"]
        parameter_groups = model.grouped_parameters(lr=lr, weight_decay=weight_decay)
        optimizer = AdamW(
            parameter_groups,
            lr=lr,
            eps=1e-8,
            betas=(1.0 - hparams["one_minus_beta1"], hparams["beta2"]),
            weight_decay=weight_decay,
            fused=False
        )
        step_hint = workload_specs.max_steps if workload_specs.max_steps else workload_specs.max_epochs
        interval = "step" if workload_specs.max_steps else "epoch"
        scheduler = cosine_warmup(step_hint, hparams["warmup_factor"], hparams["eta_min_factor"]*lr, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval
            }
        }
