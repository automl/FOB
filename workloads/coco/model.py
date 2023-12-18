from typing import Any
import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from workloads import WorkloadModel
from submissions import Submission


class COCODetectionModel(WorkloadModel):
    def __init__(self, submission: Submission):
        model = fasterrcnn_mobilenet_v3_large_fpn()
        super().__init__(model, submission)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return torch.rand(batch.size())

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def get_specs(self) -> dict[str, Any]:
        return {"max_epochs": 1}
