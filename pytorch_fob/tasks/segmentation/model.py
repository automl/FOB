import logging
import torch
from torch.nn.functional import interpolate
from mmseg.evaluation.metrics import IoUMetric
from mmengine.logging import MMLogger
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from pytorch_fob.tasks import TaskModel
from pytorch_fob.engine.parameter_groups import GroupedModel, ParameterGroup, wd_group_named_parameters, merge_parameter_splits
from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.optimizers import Optimizer
from .segformer_contiguous import SegformerForSemanticSegmentationContiguous


class SegFormerGroupedModel(GroupedModel):
    def __init__(self, model: SegformerForSemanticSegmentation | SegformerForSemanticSegmentationContiguous) -> None:
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
        model_name = config.model.name
        model_config = SegformerConfig.from_pretrained(
            model_name,
            cache_dir=config.data_dir,
            id2label=id2label,
            label2id=label2id
        )
        if config.model.contiguous_memory:
            model_class = SegformerForSemanticSegmentationContiguous
        else:
            model_class = SegformerForSemanticSegmentation
        if config.model.use_pretrained_model:
            model = model_class.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512",
                cache_dir=config.data_dir
            )
        else:
            model = model_class.from_pretrained(
                model_name,
                cache_dir=config.data_dir,
                config=model_config
            )
        super().__init__(SegFormerGroupedModel(model), optimizer, config)  # type:ignore
        self.metric = self._get_metric()

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
        ).argmax(dim=1).detach().cpu().split(logits.size(0))
        labels = labels.detach().cpu().split(logits.size(0))
        samples = [{
            'pred_sem_seg': {'data': pred},
            'gt_sem_seg': {'data': label},
        } for pred, label in zip(preds, labels)]
        self.metric.process({}, samples)

    def _reset_metric(self):
        self.metric = self._get_metric()

    def _get_metric(self):
        metric = IoUMetric()
        logger: MMLogger = MMLogger.get_current_instance()
        logger.setLevel(logging.WARN)
        metric.dataset_meta = {"classes": list(range(150))}
        return metric

    def _compute_and_log_metrics(self, stage: str):
        metrics = self.metric.compute_metrics(self.metric.results)
        for k, v in metrics.items():
            self.log(f"{stage}_{k}", v, sync_dist=True)
        self._reset_metric()

    def on_validation_start(self) -> None:
        self._reset_metric()

    def on_validation_epoch_end(self) -> None:
        self._compute_and_log_metrics("val")

    def on_test_start(self) -> None:
        self._reset_metric()

    def on_test_epoch_end(self) -> None:
        self._compute_and_log_metrics("test")
