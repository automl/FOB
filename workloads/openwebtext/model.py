# https://huggingface.co/docs/transformers/model_doc/gpt2

from typing import Any
import torch

from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel

from workloads import WorkloadModel
from runtime.specs import RuntimeSpecs
from submissions import Submission

class OpenWebTextModel(WorkloadModel):
    def __init__(self, submission: Submission):
        
        # TODO: decide which model to use; depends on the task of the model?

        mode = 3
        if mode == 1:
            # (1)
            # Initializing a model (with random weights) from the configuration
            configuration = GPT2Config()
            model = GPT2Model(configuration)
        elif mode == 2:
            # (2)
            # The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.
            model = GPT2Model.from_pretrained("gpt2")
        else:
            # (3)
            # The GPT2 Model transformer with a language modeling head on top
            # (linear layer with weights tied to the input embeddings).
            model = GPT2LMHeadModel.from_pretrained("gpt2")

        super().__init__(model, submission)
        # self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        self.model.train()
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # Ignore loss for padding tokens
        outputs = self(input_ids)
        loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        self.model.eval()
        raise NotImplementedError

    def get_specs(self) -> RuntimeSpecs:
        raise NotImplementedError
