from pathlib import Path
import evaluate
import numpy as np
import torch
from torch.nn.functional import interpolate
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from workloads import WorkloadModel
from runtime.parameter_groups import GroupedModel, ParameterGroup, wd_group_named_parameters, merge_parameter_splits
from runtime.configs import WorkloadConfig
from submissions import Submission


class SegFormerGroupedModel(GroupedModel):
    def __init__(self, model: SegformerForSemanticSegmentation) -> None:
        super().__init__(model)

    def parameter_groups(self) -> list[ParameterGroup]:
        default_params = ParameterGroup(
            named_parameters=dict(np for np in self.model.named_parameters() if np[0].startswith("segformer"))
        )
        decoder_params = ParameterGroup(
            named_parameters=dict(np for np in self.model.named_parameters() if np[0].startswith("decode_head")),
            lr_multiplier=10.
        )
        assert len(default_params) + len(decoder_params) == len(list(self.model.named_parameters()))
        split1 = [default_params, decoder_params]
        split2 = wd_group_named_parameters(self.model)
        return merge_parameter_splits(split1, split2)


class SegmentationModel(WorkloadModel):
    """
    Lightning Module for SceneParse150 semantic segmentation task.
    Model architecture used is SegFormer from https://arxiv.org/abs/2105.15203.
    Implementation inspired by 🤗 examples and tutorials:
    - https://github.com/huggingface/transformers/tree/main/examples/pytorch/semantic-segmentation
    - https://huggingface.co/blog/fine-tune-segformer
    This workload reaches a performance similar to this pretrained model:
    https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512
    """
    def __init__(self, submission: Submission, data_dir: Path, id2label: dict[int, str], label2id: dict[str, int], workload_config: WorkloadConfig):
        model_name = "nvidia/mit-b0"
        config = SegformerConfig.from_pretrained(
            model_name,
            cache_dir=data_dir,
            id2label=id2label,
            label2id=label2id
        )
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            cache_dir=data_dir,
            config=config
        )
        # model = SegformerForSemanticSegmentation.from_pretrained(
        #     "nvidia/segformer-b0-finetuned-ade-512-512",
        #     cache_dir=data_dir
        # )
        super().__init__(SegFormerGroupedModel(model), submission, workload_config)
        self.metric = self._get_metric(data_dir)
        self._reset_metrics()

    def forward(self, x):
        return self.model(**x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        outputs = self.forward(batch)
        loss = outputs.loss
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        self._process_batch_metric(batch["labels"], outputs.logits)
        self.log("val_loss", outputs.loss, sync_dist=True)

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
        # currently compute is very slow, using _compute instead
        # see: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
        metrics = self.metric._compute(
            predictions=preds,
            references=labels,
            num_labels=150,
            ignore_index=255,
            reduce_labels=False
        )
        if metrics is not None:
            for metric in metrics.keys():
                if metric.startswith("mean"):
                    self.metrics[metric].append(metrics[metric])

    def _reset_metrics(self):
        self.metrics = {
            "mean_iou": [],
            "mean_accuracy": []
        }

    def _get_metric(self, data_dir: Path):
        specs = self.get_specs()
        if isinstance(specs.devices, int):
            num_process = specs.devices
        elif isinstance(specs.devices, list):
            num_process = len(specs.devices)
        else:
            raise TypeError(f"could not infer num_process from {specs.devices=}")
        return evaluate.load("mean_iou", num_process=num_process, cache_dir=str(data_dir))

    def _compute_and_log_metrics(self, stage: str):
        for k, v in self.metrics.items():
            self.log(f"{stage}_{k}", np.mean(v), sync_dist=True)  # type:ignore
        self._reset_metrics()

    def on_validation_epoch_end(self) -> None:
        self._compute_and_log_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._compute_and_log_metrics("test")
