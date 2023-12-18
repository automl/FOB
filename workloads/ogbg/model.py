from typing import Any
import torch
from workloads import WorkloadModel
from submissions import Submission

class OGBGModel(WorkloadModel):
    def __init__(self, submission: Submission):
        
        # TODO: decide which model to use; depends on the task of the model?
        model = None
        super().__init__(model, submission)

        # datasets:
        #   ogbg-molhiv
        #   https://ogb.stanford.edu/docs/graphprop/
        # task:
        #   graph property prediction
        # metric:
        #   ROC-AUC
        # self.loss_fn = None # TODO

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def get_specs(self) -> dict[str, Any]:
        raise NotImplementedError
