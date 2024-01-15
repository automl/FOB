from lightning import Callback, Trainer
from workloads.coco.model import COCODetectionModel


class COCOEvalSummarize(Callback):
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: COCODetectionModel):
        self._summarize_eval(pl_module)
        pl_module.log("val_AP", pl_module.get_eval_stats()[0], sync_dist=True)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: COCODetectionModel):
        self._summarize_eval(pl_module)
        pl_module.log("test_AP", pl_module.get_eval_stats()[0], sync_dist=True)

    def _summarize_eval(self, pl_module: COCODetectionModel):
        pl_module.coco_eval.synchronize_between_processes()
        pl_module.coco_eval.accumulate()
        pl_module.coco_eval.summarize()

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: COCODetectionModel):
        pl_module.reset_coco_eval()

    def on_test_epoch_start(self, trainer: Trainer, pl_module: COCODetectionModel):
        pl_module.reset_coco_eval()
