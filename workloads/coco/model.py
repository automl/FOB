from typing import Any
import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from workloads import WorkloadModel
from submissions import Submission


class COCODetectionModel(WorkloadModel):
    def __init__(self, submission: Submission):
        model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=91)
        super().__init__(model, submission)

    def forward(self, x):
        imgs, targets = x
        return self.model(imgs, targets)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        loss_dict = self.model(imgs, targets)
        self.log_losses(loss_dict, "train")
        return self.total_loss(loss_dict)

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        self.model(imgs, targets)
        # TODO: calculate metrics

    def test_step(self, batch, batch_idx):
        # no test labels available for coco
        pass

    def log_losses(self, losses: dict, stage: str):
        for loss, val in losses.items():
            self.log(f"{stage}_{loss}", val)
        total = sum(loss for loss in losses.values())
        self.log(f"{stage}_loss_total", total)

    def total_loss(self, losses: dict):
        return sum(loss for loss in losses.values())

    def get_specs(self) -> dict[str, Any]:
        return {"max_epochs": 10}
