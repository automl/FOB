# https://huggingface.co/docs/transformers/model_doc/gpt2

from typing import Any
import torch

from transformers import AutoTokenizer, GPT2Config, GPT2Model, GPT2LMHeadModel

from workloads import WorkloadModel
from submissions import Submission

class OpenWebTextModel(WorkloadModel):
    def __init__(self, submission: Submission):
        
        # TODO: decide which model to use; depends on the task of the model?

        # (1)
        # model = densenet121()
        # Initializing a model (with random weights) from the configuration
        configuration = GPT2Config()
        model = GPT2Model(configuration)
        
        # (2)
        # The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = GPT2Model.from_pretrained("gpt2")

        # (3)
        # The GPT2 Model transformer with a language modeling head on top
        # (linear layer with weights tied to the input embeddings).
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")

        super().__init__(model, submission)
        # self.loss_fn = nn.CrossEntropyLoss()

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
