from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import SGD
from runtime.parameter_groups import GroupedModel
from runtime.configs import SubmissionConfig, WorkloadConfig


def configure_optimizers(
        model: GroupedModel,
        workload_config: WorkloadConfig,
        submission_config: SubmissionConfig
    ) -> OptimizerLRScheduler:
    hparams = submission_config
    lr = hparams["learning_rate"]
    return {"optimizer": SGD(model.grouped_parameters(lr=lr), lr=lr)}
