from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models import MobileNet_V3_Large_Weights
from pycocotools.coco import COCO
from workloads import WorkloadModel
from workloads.specs import RuntimeSpecs
from submissions import Submission
from .coco_eval import CocoEvaluator


class COCODetectionModel(WorkloadModel):
    """
    Lightning Module for COCO object detection task.
    Implementation is heavily inspired by
    https://github.com/pytorch/vision/tree/main/references/detection
    """
    def __init__(self, submission: Submission, eval_gts: COCO):
        model = fasterrcnn_mobilenet_v3_large_fpn(
            num_classes=91,
            weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1
        )
        super().__init__(model, submission)
        self.eval_gts = eval_gts
        self.reset_coco_eval()

    def forward(self, x):
        imgs, targets = x
        return self.model(imgs, targets)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        loss_dict = self.model(imgs, targets)
        self.log_losses(loss_dict, "train")
        return self.total_loss(loss_dict)

    def validation_step(self, batch, batch_idx):
        self._update_coco_eval(batch)

    def test_step(self, batch, batch_idx):
        self._update_coco_eval(batch)

    def reset_coco_eval(self):
        self.coco_eval = CocoEvaluator(self.eval_gts, ["bbox"])

    def _update_coco_eval(self, batch):
        imgs, targets = batch
        preds = self.model(imgs)
        res = {target["image_id"]: pred for target, pred in zip(targets, preds)}
        self.coco_eval.update(res)

    def get_eval_stats(self) -> list[float]:
        return self.coco_eval.coco_eval["bbox"].stats

    def log_losses(self, losses: dict, stage: str):
        for loss, val in losses.items():
            self.log(f"{stage}_{loss}", val)
        total = sum(loss for loss in losses.values())
        self.log(f"{stage}_loss_total", total)

    def total_loss(self, losses: dict):
        return sum(loss for loss in losses.values())

    def get_specs(self) -> RuntimeSpecs:
        return RuntimeSpecs(
            max_epochs=26,
            max_steps=381_134,
            devices=1,  # TODO: correct devices (4)
            target_metric="val_AP",
            target_metric_mode="max"
        )
