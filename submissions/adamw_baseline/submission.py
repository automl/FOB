from typing import Optional
from pathlib import Path
from torch.nn import Module
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from workloads.specs import SubmissionSpecs
from submissions import Submission
from bob.runtime import RuntimeArgs


def get_submission(runtime_args: RuntimeArgs) -> Submission:
    return AdamWBaseline(runtime_args.hyperparameter_path)


def cosine_warmup(step_hint: int, warmup_factor: float, optimizer):
    warmup_steps = round(warmup_factor * step_hint)
    warmup = LinearLR(
        optimizer, start_factor=1e-10, end_factor=1., total_iters=warmup_steps)
    cosine_steps = max(step_hint - warmup_steps, 1)
    cosine_decay = CosineAnnealingLR(optimizer, T_max=cosine_steps)
    return SequentialLR(
        optimizer, schedulers=[warmup, cosine_decay], milestones=[warmup_steps])


class AdamWBaseline(Submission):
    def __init__(self, hyperparameter_path: Optional[Path] = None) -> None:
        if hyperparameter_path is None:
            hyperparameter_path = Path(__file__).parent.joinpath("hyperparameters.json")
        super().__init__(hyperparameter_path)

    def configure_optimizers(self, model: Module, workload_specs: SubmissionSpecs) -> OptimizerLRScheduler:
        hparams = self.get_hyperparameters()
        optimizer = AdamW(
            model.parameters(),
            lr=hparams["learning_rate"],
            eps=1e-8,
            betas=(1.0 - hparams["one_minus_beta1"], hparams["beta2"]),
            weight_decay=hparams["weight_decay"],
            fused=False
        )
        step_hint = workload_specs.max_steps if workload_specs.max_steps else workload_specs.max_epochs
        interval = "step" if workload_specs.max_steps else "epoch"
        scheduler = cosine_warmup(step_hint, hparams["warmup_factor"], optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval
            }
        }
