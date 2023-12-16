from typing import Any
import torch
from workloads import WorkloadModel
from submissions import Submission


class MNISTModel(WorkloadModel):
    def __init__(self, submission: Submission):

        input_size = 28 * 28  # 784
        num_hidden = 128
        num_classes = 10

        # algoperf net
        # https://github.com/mlcommons/algorithmic-efficiency/blob/main/algorithmic_efficiency/workloads/mnist/mnist_pytorch/workload.py
        model = torch.nn.Sequential(
            torch.nn.Linear(input_size, num_hidden, bias=True),
            torch.nn.Sigmoid(),
            torch.nn.Linear(num_hidden, num_classes, bias=True),
        )
        super().__init__(model, submission)
        # negative log likelihood loss
        self.loss = torch.nn.functional.nll_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        output = self.model(x)
        prediction = torch.softmax(output, dim=1)
        return prediction

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)

        # # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.compute_and_log_loss(batch, "val_loss")

    def test_step(self, batch, batch_idx):
        self.compute_and_log_loss(batch, "test_loss")

    def compute_and_log_loss(self, batch, log_name: str):
        x, y = batch
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log(log_name, loss)

    def get_specs(self) -> dict[str, Any]:
        return {"max_epochs": 10}
