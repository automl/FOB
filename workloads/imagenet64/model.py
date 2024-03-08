import numpy as np
import torch
from torch import nn
from timm import create_model
from sklearn.metrics import top_k_accuracy_score
from workloads import WorkloadModel
from runtime.configs import WorkloadConfig
from submissions import Submission


class ImagenetModel(WorkloadModel):
    def __init__(self, submission: Submission, workload_config: WorkloadConfig):
        model_name = workload_config.model.name
        model = create_model(model_name)
        
        # 7x7 conv might be pretty large for 32x32 images
        model.conv1 = nn.Conv2d(3,  # rgb color
                                workload_config.model.hidden_channel,
                                kernel_size=workload_config.model.kernel_size,
                                padding=workload_config.model.padding,
                                bias=False
                                )

        # pooling small images might be bad
        if not workload_config.model.maxpool:
            model.maxpool = nn.Identity()  # type:ignore

        super().__init__(model, submission, workload_config)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        imgs, labels = batch["image"], batch["label"]
        return self.model(imgs), labels

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        preds, labels = self.forward(batch)
        loss = self.compute_and_log_loss(preds, labels, "train")
        self.compute_and_log_acc(preds, labels, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        preds, labels = self.forward(batch)
        self.compute_and_log_loss(preds, labels, "val")
        self.compute_and_log_acc(preds, labels, "val")

    def test_step(self, batch, batch_idx):
        preds, labels = self.forward(batch)
        self.compute_and_log_acc(preds, labels, "test")

    def compute_and_log_acc(self, preds: torch.Tensor, labels: torch.Tensor, stage: str) -> dict[str, float]:
        pred_probs = preds.softmax(-1).detach().cpu().numpy()
        gts = labels.detach().cpu().numpy()
        all_labels = np.arange(1000)
        top_1_acc = top_k_accuracy_score(y_true=gts, y_score=pred_probs, k=1, labels=all_labels)
        top_5_acc = top_k_accuracy_score(y_true=gts, y_score=pred_probs, k=5, labels=all_labels)
        self.log(f"{stage}_top1_acc", top_1_acc, sync_dist=True)
        self.log(f"{stage}_top1_err", 1 - top_1_acc, sync_dist=True)
        self.log(f"{stage}_top5_acc", top_5_acc, sync_dist=True)
        self.log(f"{stage}_top5_err", 1 - top_5_acc, sync_dist=True)
        return {"top1": top_1_acc, "top5": top_5_acc}

    def compute_and_log_loss(self, preds: torch.Tensor, labels: torch.Tensor, stage: str) -> torch.Tensor:
        loss = self.loss_fn(preds, labels)
        self.log(f"{stage}_loss", loss, sync_dist=True)
        return loss
