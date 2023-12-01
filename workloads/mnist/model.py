from typing import Callable
from lightning import LightningModule
import torch

class MNISTModel(LightningModule):
    def __init__(self, create_optimizer_fn: Callable):
        super().__init__()
        self.create_optimizer_fn = create_optimizer_fn

        input_size = 28 * 28  # 784
        num_hidden = 128
        num_classes = 10
        
        # TODO param?
        lr_rate = 1.0
        self.lr_rate = lr_rate
        
        # algoperf net
        # https://github.com/mlcommons/algorithmic-efficiency/blob/main/algorithmic_efficiency/workloads/mnist/mnist_pytorch/workload.py
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, num_hidden, bias=True),
            torch.nn.Sigmoid(),
            torch.nn.Linear(num_hidden, num_classes, bias=True),
        )
        # negative log likelihood loss
        self.loss = torch.nn.functional.nll_loss

    def forward(self, x):
        batch_size, channels, widht, height = x.size()
        
        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        output = self.model(x)
        prediction = torch.softmax(output, dim=1)
        return prediction


    def training_step(self, batch, batch_idx):
        x, y = batch 
        batch_size, channels, widht, height = x.size()
        # import pdb; pdb.set_trace()
        x = x.view(batch_size, -1)
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}
    
        # # Logging to TensorBoard (if installed) by default
        # self.log("train_loss", loss)
        # return loss

    def configure_optimizers(self):
        return self.create_optimizer_fn(self)
