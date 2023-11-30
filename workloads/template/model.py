from typing import Callable
from lightning import LightningModule
import torch

class TemplateModel(LightningModule):
    def __init__(self, create_optimizer_fn: Callable):
        super().__init__()
        self.create_optimizer_fn = create_optimizer_fn
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
            torch.nn.ReLU(),
        )

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.create_optimizer_fn(self)
