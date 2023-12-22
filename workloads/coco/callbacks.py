from lightning import Callback
from workloads.coco.model import COCODetectionModel

class COCOEvalSummarize(Callback):
    def on_validation_epoch_end(self, trainer, pl_module: COCODetectionModel):
        pl_module.coco_eval.accumulate()
        pl_module.coco_eval.summarize()
        pl_module.log("val_AP", pl_module.coco_eval.stats[0])

    def on_test_epoch_end(self, trainer, pl_module: COCODetectionModel):
        pl_module.coco_eval.accumulate()
        pl_module.coco_eval.summarize()
        pl_module.log("test_AP", pl_module.coco_eval.stats[0])

    def on_validation_epoch_start(self, trainer, pl_module: COCODetectionModel):
        pl_module.reset_coco_eval()

    def on_test_epoch_start(self, trainer, pl_module: COCODetectionModel):
        pl_module.reset_coco_eval()
