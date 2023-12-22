from typing import Any
from contextlib import redirect_stdout
import io
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models import MobileNet_V3_Large_Weights
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from workloads import WorkloadModel
from submissions import Submission


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
        imgs, targets = batch
        preds = self.model(imgs)
        res = {target["image_id"]: pred for target, pred in zip(targets, preds)}
        self._update_eval(res)

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def reset_coco_eval(self):
        self.coco_eval = COCOeval(cocoGt=self.eval_gts, iouType="bbox")
        self.img_ids = []
        self.eval_imgs = []

    def _update_eval(self, predictions: dict):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)
        results = self._prepare_for_coco_detection(predictions)
        with redirect_stdout(io.StringIO()):
            coco_dt = COCO.loadRes(self.eval_gts, results) if results else COCO()  # type:ignore (this is fine)
        self.coco_eval.cocoDt = coco_dt
        self.coco_eval.params.imgIds = list(img_ids)
        with redirect_stdout(io.StringIO()):
            self.coco_eval.evaluate()
        eval_imgs = np.asarray(self.coco_eval.evalImgs).reshape(
            -1,
            len(self.coco_eval.params.areaRng),
            len(self.coco_eval.params.imgIds)
        )
        self.eval_imgs.append(eval_imgs)
        return eval_imgs

    def _prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = self._convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results


    def _convert_to_xywh(self, boxes):
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

    def log_losses(self, losses: dict, stage: str):
        for loss, val in losses.items():
            self.log(f"{stage}_{loss}", val)
        total = sum(loss for loss in losses.values())
        self.log(f"{stage}_loss_total", total)

    def total_loss(self, losses: dict):
        return sum(loss for loss in losses.values())

    def get_specs(self) -> dict[str, Any]:
        return {"max_epochs": 26}
