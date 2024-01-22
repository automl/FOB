from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import SGD
from submissions import Submission
from runtime.parameter_groups import GroupedModel
from runtime.specs import SubmissionSpecs
from runtime import RuntimeArgs


def get_submission(runtime_args: RuntimeArgs) -> Submission:
    return TemplateSubmission(runtime_args.hyperparameter_path)


class TemplateSubmission(Submission):

    def configure_optimizers(self, model: GroupedModel, workload_specs: SubmissionSpecs) -> OptimizerLRScheduler:
        hparams = self.get_hyperparameters()
        return {"optimizer": SGD(model.grouped_parameters(lr=hparams["learning_rate"]), lr=hparams["learning_rate"])}
