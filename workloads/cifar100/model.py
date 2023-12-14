from typing import Callable
from lightning import LightningModule
from torchvision.models import resnet34
from torch import nn
import torch

class CIFAR100Model(LightningModule):
    def __init__(self, create_optimizer_fn: Callable):
        super().__init__()
        self.create_optimizer_fn = create_optimizer_fn
        self.model = resnet34()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_fn(preds, labels)
        self.compute_and_log_acc(preds, labels, "train_acc")
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        self.compute_and_log_acc(preds, labels, "val_acc")

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        self.compute_and_log_acc(preds, labels, "test_acc")

    def compute_and_log_acc(self, preds: torch.Tensor, labels: torch.Tensor, log_label: str):
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log(log_label, acc)

    def configure_optimizers(self):
        return self.create_optimizer_fn(self)
