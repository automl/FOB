from pathlib import Path
import evaluate
import numpy as np
import torch
from torch.nn.functional import interpolate
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from tasks import TaskModel
from engine.parameter_groups import GroupedModel, ParameterGroup, wd_group_named_parameters, merge_parameter_splits
from engine.configs import TaskConfig
from optimizers import Optimizer


class SegFormerGroupedModel(GroupedModel):
    def __init__(self, model: SegformerForSemanticSegmentation) -> None:
        super().__init__(model)

    def parameter_groups(self) -> list[ParameterGroup]:
        backbone_params = ParameterGroup(
            named_parameters=dict(np for np in self.model.named_parameters() if np[0].startswith("segformer")),
            lr_multiplier=0.1
        )
        decoder_params = ParameterGroup(
            named_parameters=dict(np for np in self.model.named_parameters() if np[0].startswith("decode_head"))
        )
        assert len(backbone_params) + len(decoder_params) == len(list(self.model.named_parameters()))
        split1 = [backbone_params, decoder_params]
        split2 = wd_group_named_parameters(self.model)
        return merge_parameter_splits(split1, split2)


class SegmentationModel(TaskModel):
    """
    Lightning Module for SceneParse150 semantic segmentation task.
    Model architecture used is SegFormer from https://arxiv.org/abs/2105.15203.
    Implementation inspired by 🤗 examples and tutorials:
    - https://github.com/huggingface/transformers/tree/main/examples/pytorch/semantic-segmentation
    - https://huggingface.co/blog/fine-tune-segformer
    This task reaches a performance similar to this pretrained model:
    https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512
    """
    def __init__(
            self,
            optimizer: Optimizer,
            id2label: dict[int, str],
            label2id: dict[str, int],
            config: TaskConfig
        ):
        model_name = "nvidia/mit-b0"
        model_config = SegformerConfig.from_pretrained(
            model_name,
            cache_dir=config.data_dir,
            id2label=id2label,
            label2id=label2id
        )
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            cache_dir=config.data_dir,
            config=model_config
        )
        # model = SegformerForSemanticSegmentation.from_pretrained(
        #     "nvidia/segformer-b0-finetuned-ade-512-512",
        #     cache_dir=data_dir
        # )
        super().__init__(SegFormerGroupedModel(model), optimizer, config)
        self.metric = self._get_metric()
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

    def _get_metric(self, num_process: int = 4):
        # TODO: get num_process from devices
        return evaluate.load(
            "mean_iou",
            num_process=num_process,
            cache_dir=str(self.config.data_dir)
        )

    def _compute_and_log_metrics(self, stage: str):
        for k, v in self.metrics.items():
            self.log(f"{stage}_{k}", np.mean(v), sync_dist=True)  # type:ignore
        self._reset_metrics()

    def on_validation_epoch_end(self) -> None:
        self._compute_and_log_metrics("val")

    def on_test_start(self) -> None:
        self.metric = self._get_metric(num_process=1)

    def on_test_epoch_end(self) -> None:
        self._compute_and_log_metrics("test")
