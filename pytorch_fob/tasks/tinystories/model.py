import torch

from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.optimizers import Optimizer
from pytorch_fob.tasks import TaskModel

from .nanogpt import GPT, GPTConfig


class GPTModel(TaskModel):
    def __init__(self, optimizer: Optimizer, config: TaskConfig, vocab_size=65):
        gptcfg = GPTConfig(
            n_embd=config.model.n_embd,
            n_head=config.model.n_head,
            n_layer=config.model.n_layer,
            block_size=config.model.block_size,
            vocab_size=vocab_size,
            dropout=config.model.dropout,
            bias=config.model.bias,
        )

        model = GPT(gptcfg)

        super().__init__(model, optimizer, config)

    def forward(self, batch):
        idx, targets = batch
        return self.model.forward(idx, targets)

    def training_step(self, batch, batch_idx):
        return self.compute_and_log_metrics(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.compute_and_log_metrics(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.compute_and_log_metrics(batch, "test")

    def compute_and_log_metrics(self, batch, prefix: str):
        logits, loss = self.forward(batch)

        self.log(f"{prefix}_loss", loss, sync_dist=True)

        # Calculate perplexity manually
        # Perplexity = exp(average negative log likelihood) = exp(cross entropy loss)
        with torch.no_grad():
            perplexity = torch.exp(loss)

        self.log(f"{prefix}_perplexity", perplexity.item(), sync_dist=True)

        return loss
