import torch
from transformers import AutoConfig, AutoModelForSemanticSegmentation
from workloads import WorkloadModel
from runtime.specs import RuntimeSpecs
from submissions import Submission


class SegmentationModel(WorkloadModel):
    """
    Lightning Module for SceneParse150 semantic segmentation task.
    Model architecture used is SegFormer from https://arxiv.org/abs/2105.15203.
    """
    def __init__(self, submission: Submission):
        config = AutoConfig.from_pretrained("nvidia/mit-b0")
        model = AutoModelForSemanticSegmentation.from_config(config)
        super().__init__(model, submission)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(**x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        outputs = self.forward(batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = outputs.loss
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = outputs.loss
        self.log("test_loss", loss)

    def compute_and_log_acc(self, preds: torch.Tensor, targets: torch.Tensor, log_label: str) -> torch.Tensor:
        raise NotImplementedError("TODO")

    def compute_and_log_loss(self, preds: torch.Tensor, targets: torch.Tensor, log_label: str) -> torch.Tensor:
        loss = self.loss_fn(preds, targets)
        self.log(log_label, loss)
        return loss

    def get_specs(self) -> RuntimeSpecs:
        return RuntimeSpecs(
            max_epochs=26,
            max_steps=None,
            devices=1,  # TODO: correct devices (4)
            target_metric="val_loss",  # TODO: correct metric (IoU)
            target_metric_mode="min"
        )
