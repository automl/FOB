import torch
from pytorch_fob.tasks import TaskModel
from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.optimizers import Optimizer


class MNISTModel(TaskModel):
    def __init__(self, optimizer: Optimizer, config: TaskConfig):

        input_size = 28 * 28  # 784
        num_classes = 10
        num_hidden = config.model.num_hidden
        activation = config.model.activation
        if activation.lower() == "Sigmoid".lower():
            self.activation = torch.nn.Sigmoid
        elif activation.lower() == "ReLU".lower():
            self.activation = torch.nn.ReLU
        else:
            raise NotImplementedError(f"{activation} is not supported for mnist yet")

        # algoperf net
        # https://github.com/mlcommons/algorithmic-efficiency/blob/main/algorithmic_efficiency/workloads/mnist/mnist_pytorch/workload.py
        model = torch.nn.Sequential(
            torch.nn.Linear(input_size, num_hidden, bias=True),
            self.activation(),
            torch.nn.Linear(num_hidden, num_classes, bias=True),
        )
        super().__init__(model, optimizer, config)
        # negative log likelihood loss
        self.loss_fn = torch.nn.functional.nll_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        output = self.model(x)
        prediction = torch.softmax(output, dim=1)
        return prediction

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.compute_and_log_loss(y_hat, y, "train_loss")
        self.compute_and_log_acc(y_hat, y, "train_acc")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        self.compute_and_log_loss(y_hat, y, "val_loss")
        self.compute_and_log_acc(y_hat, y, "val_acc")

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        self.compute_and_log_acc(y_hat, y, "test_acc")

    def compute_and_log_loss(self, preds: torch.Tensor, labels: torch.Tensor, log_name: str) -> torch.Tensor:
        loss = self.loss_fn(preds, labels)
        self.log(log_name, loss)
        return loss

    def compute_and_log_acc(self, preds: torch.Tensor, labels: torch.Tensor, log_label: str) -> torch.Tensor:
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log(log_label, acc)
        return acc
