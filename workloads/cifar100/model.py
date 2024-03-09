import torch
from torch import nn
from torchvision.models import resnet18
from workloads import WorkloadModel
from engine.configs import WorkloadConfig
from optimizers import Optimizer


class CIFAR100Model(WorkloadModel):
    def __init__(self, optimizer: Optimizer, workload_config: WorkloadConfig):
        model = resnet18(num_classes=100, weights=None)
        # 7x7 conv is too large for 32x32 images
        model.conv1 = nn.Conv2d(3,  # rgb color
                                workload_config.model.hidden_channel,
                                kernel_size=workload_config.model.kernel_size,
                                padding=workload_config.model.padding,
                                bias=False
                                )

        # pooling small images is bad
        if not workload_config.model.maxpool:
            model.maxpool = nn.Identity()  # type:ignore
        super().__init__(model, optimizer, workload_config)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.compute_and_log_loss(preds, labels, "train_loss")
        self.compute_and_log_acc(preds, labels, "train_acc")
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        self.compute_and_log_loss(preds, labels, "val_loss")
        self.compute_and_log_acc(preds, labels, "val_acc")

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        self.compute_and_log_acc(preds, labels, "test_acc")

    def compute_and_log_acc(self, preds: torch.Tensor, labels: torch.Tensor, log_label: str) -> torch.Tensor:
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log(log_label, acc)
        return acc

    def compute_and_log_loss(self, preds: torch.Tensor, labels: torch.Tensor, log_label: str) -> torch.Tensor:
        loss = self.loss_fn(preds, labels)
        self.log(log_label, loss)
        return loss
