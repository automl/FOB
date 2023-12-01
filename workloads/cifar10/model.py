from typing import Callable
from lightning import LightningModule
import torch

class CIFAR10Model(LightningModule):
    def __init__(self, create_optimizer_fn: Callable):
        super().__init__()
        self.create_optimizer_fn = create_optimizer_fn
        
        self.norm_layer = torch.nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError


    def configure_optimizers(self):
        return self.create_optimizer_fn(self)
