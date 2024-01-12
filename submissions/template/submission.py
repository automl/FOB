from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.nn import Module
from torch.optim import SGD
from submissions import Submission
from runtime.specs import SubmissionSpecs
from runtime import RuntimeArgs


def get_submission(runtime_args: RuntimeArgs) -> Submission:
    return TemplateSubmission(runtime_args.hyperparameter_path)


class TemplateSubmission(Submission):

    def configure_optimizers(self, model: Module, workload_specs: SubmissionSpecs) -> OptimizerLRScheduler:
        hparams = self.get_hyperparameters()
        return {"optimizer": SGD(model.parameters(), lr=hparams["learning_rate"])}
