import evaluate
import torch
from torch.nn.functional import interpolate
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
        self._reset_metric()

    def forward(self, x):
        return self.model(**x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        outputs = self.forward(batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        self._process_batch_metric(batch["labels"], outputs.logits)
        self.log("val_loss", outputs.loss)

    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        self._process_batch_metric(batch["labels"], outputs.logits)

    def _process_batch_metric(self, labels: torch.Tensor, logits: torch.Tensor):
        preds = interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False
        ).argmax(dim=1)
        self.metric.add_batch(predictions=preds, references=labels)

    def _reset_metric(self):
        self.metric = evaluate.load("mean_iou")

    def get_specs(self) -> RuntimeSpecs:
        return RuntimeSpecs(
            max_epochs=26,
            max_steps=None,
            devices=1,  # TODO: correct devices (4)
            target_metric="val_mean_accuracy",
            target_metric_mode="max"
        )

    def _compute_and_log_metrics(self, stage: str):
        metrics = self.metric.compute(num_labels=150, ignore_index=255, reduce_labels=False)
        if not metrics is None:
            for metric in metrics.keys():
                if metric.startswith("mean"):
                    self.log(f"{stage}_{metric}", metrics[metric])
        self._reset_metric()

    def on_validation_epoch_end(self) -> None:
        self._compute_and_log_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._compute_and_log_metrics("test")
