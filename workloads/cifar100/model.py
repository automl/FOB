from typing import Any
import torch
from torch import nn
from torchvision.models import resnet18
from workloads import WorkloadModel
from submissions import Submission


class CIFAR100Model(WorkloadModel):
    def __init__(self, submission: Submission):
        model = resnet18(num_classes=100, weights=None)
        # 7x7 conv is too large for 32x32 images
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        # pooling small images is bad
        model.maxpool = nn.Identity()  #type:ignore
        super().__init__(model, submission)
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

    def get_specs(self) -> dict[str, Any]:
        return {"max_epochs": 100}
