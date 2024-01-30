from typing import Any
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import SGD
from submissions import Submission
from runtime.parameter_groups import GroupedModel
from runtime.specs import SubmissionSpecs


def get_submission(hyperparameters: dict[str, Any]) -> Submission:
    return TemplateSubmission(hyperparameters)


class TemplateSubmission(Submission):

    def configure_optimizers(self, model: GroupedModel, workload_specs: SubmissionSpecs) -> OptimizerLRScheduler:
        hparams = self.get_hyperparameters()
        return {"optimizer": SGD(model.grouped_parameters(lr=hparams["learning_rate"]), lr=hparams["learning_rate"])}
