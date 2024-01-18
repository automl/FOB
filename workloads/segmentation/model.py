from pathlib import Path
import evaluate
import numpy as np
import torch
from torch.nn.functional import interpolate
from transformers import SegformerForSemanticSegmentation
from workloads import WorkloadModel
from runtime.specs import RuntimeSpecs
from submissions import Submission


class SegmentationModel(WorkloadModel):
    """
    Lightning Module for SceneParse150 semantic segmentation task.
    Model architecture used is SegFormer from https://arxiv.org/abs/2105.15203.
    """
    def __init__(self, submission: Submission, cache_dir: Path):
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", cache_dir=cache_dir)
        super().__init__(model, submission)
        self.metric = self._get_metric()
        self._reset_metrics()

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
        ).argmax(dim=1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        metrics = self.metric._compute(
            predictions=preds,
            references=labels,
            num_labels=150,
            ignore_index=255,
            reduce_labels=False
        )
        if not metrics is None:
            for metric in metrics.keys():
                if metric.startswith("mean"):
                    self.metrics[metric].append(metrics[metric])


    def _reset_metrics(self):
        self.metrics = {
            "mean_iou": [],
            "mean_accuracy": []
        }

    def _get_metric(self):
        specs = self.get_specs()
        if isinstance(specs.devices, int):
            num_process = specs.devices
        elif isinstance(specs.devices, list):
            num_process = len(specs.devices)
        else:
            raise TypeError(f"could not infer num_process from {specs.devices=}")
        return evaluate.load("mean_iou", num_process=num_process)

    def get_specs(self) -> RuntimeSpecs:
        return RuntimeSpecs(
            max_epochs=50,
            max_steps=None,
            devices=1,  # TODO: correct devices (4)
            target_metric="val_mean_accuracy",
            target_metric_mode="max"
        )

    def _compute_and_log_metrics(self, stage: str):
        for k, v in self.metrics.items():
            self.log(f"{stage}_{k}", np.mean(v))
        self._reset_metrics()

    def on_validation_epoch_end(self) -> None:
        self._compute_and_log_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._compute_and_log_metrics("test")
