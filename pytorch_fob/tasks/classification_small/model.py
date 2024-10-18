import torch
from torch import nn
from torch.nn import init
from torchvision.models import resnet18

from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.optimizers import Optimizer
from pytorch_fob.tasks import TaskModel


class CIFAR100Model(TaskModel):
    def __init__(self, optimizer: Optimizer, config: TaskConfig):
        model = resnet18(num_classes=100, weights=None)
        # 7x7 conv is too large for 32x32 images
        model.conv1 = nn.Conv2d(3,  # rgb color
                                config.model.hidden_channel,
                                kernel_size=config.model.kernel_size,
                                padding=config.model.padding,
                                bias=False
                                )

        # pooling small images is bad
        if not config.model.maxpool:
            model.maxpool = nn.Identity()  # type:ignore

        super().__init__(model, optimizer, config)

        # init weights
        self._init_weights(config)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.compute_and_log_loss(preds, labels, "train_loss")
        self.compute_and_log_acc(preds, labels, "train_acc")
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        self.compute_and_log_loss(preds, labels, "val_loss")
        self.compute_and_log_acc(preds, labels, "val_acc")

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        self.compute_and_log_acc(preds, labels, "test_acc")

    def compute_and_log_acc(self, preds: torch.Tensor, labels: torch.Tensor, log_label: str) -> torch.Tensor:
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log(log_label, acc)
        return acc

    def compute_and_log_loss(self, preds: torch.Tensor, labels: torch.Tensor, log_label: str) -> torch.Tensor:
        loss = self.loss_fn(preds, labels)
        self.log(log_label, loss)
        return loss

    def _init_weights(self, config: TaskConfig):
        if config.model.init_fn is None:
            return
        fn_name = config.model.init_fn
        if hasattr(init, f"{fn_name}_"):
            init_fn = getattr(init, f"{fn_name}_")
        elif hasattr(init, fn_name):
            init_fn = getattr(init, fn_name)
        else:
            raise ValueError(f"{fn_name} is not a valid init function")
        self._init_conv_weights(init_fn, **(config.model.init_fn_kwargs or {}))

    def _init_conv_weights(self, init_fn, **kwargs):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                init_fn(m.weight, **kwargs)
