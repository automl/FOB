from typing import Callable
from lightning import LightningModule
import torch

class CIFAR100Model(LightningModule):
    def __init__(self, create_optimizer_fn: Callable):
        super().__init__()
        self.create_optimizer_fn = create_optimizer_fn
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError


    def configure_optimizers(self):
        return self.create_optimizer_fn(self)
