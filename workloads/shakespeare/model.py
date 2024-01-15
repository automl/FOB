# https://huggingface.co/docs/transformers/model_doc/gpt2

from typing import Any
import torch

from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel

from workloads import WorkloadModel
from runtime.specs import RuntimeSpecs
from submissions import Submission

class ShakespeareModel(WorkloadModel):
    def __init__(self, submission: Submission):
        
        ...

        super().__init__(model, submission)
        # self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        ...

    def validation_step(self, batch, batch_idx):
        ...
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        ...
        raise NotImplementedError

    def get_specs(self) -> RuntimeSpecs:
        raise NotImplementedError

