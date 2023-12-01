# adapted from https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/04-inception-resnet-densenet.html
from typing import Callable
from lightning import LightningModule
from torch import nn


class CIFAR10Model(LightningModule):
    def __init__(self, create_optimizer_fn: Callable):
        super().__init__()
        self.create_optimizer_fn = create_optimizer_fn

        # some parameter, maybe put in namespace? is here for now
        self.num_classes = 10,
        self.num_blocks = [3, 3, 3]
        self.c_hidden = [16, 32, 64]
        self.activation_fn = nn.ReLU(inplace=True)
        self.block = ResNetBlock
        
        # INPUT net
        # A first convolution on the original image to scale up the channel size
        self.input_net = nn.Sequential(
                nn.Conv2d(3, self.c_hidden[0], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.c_hidden[0]),
                self.activation_fn(),
            )

        # BLOCKS net
        # Creating the ResNet blocks
        blocks = []
        for block_id, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group, except the very first one.
                subsample = block_id > 0 and (bc == 0)
                block = self.block(
                    c_in=self.c_hidden[block_id if not subsample else (block_id - 1)],
                    act_fn=self.activation_fn,
                    subsample=subsample,
                    c_out=self.c_hidden[block_id]
                )
                blocks.append(block)
        self.blocks = nn.Sequential(*blocks)

        # OUTPUT net
        # Mapping to classification output
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.c_hidden[-1], self.num_classes)
        )

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.output_net(x)
        return x

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss  # Return tensor to call ".backward" on


    def configure_optimizers(self):
        return self.create_optimizer_fn(self)


# https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/04-inception-resnet-densenet.html#ResNet
class ResNetBlock(nn.Module):
    def __init__(self, c_in, act_fn, subsample=False, c_out=-1):
        """ResNetBlock.

        Args:
            c_in: Number of input features
            act_fn: Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample:
            c_out = c_in

        # Network representing F
        self.net = nn.Sequential(
            nn.Conv2d(
                c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False
            ),  # No bias needed as the Batch Norm handles it
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )

        # 1x1 convolution with stride 2 means we take the upper left value, and transform it to new output size
        self.downsample = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2) if subsample else None
        self.act_fn = act_fn()

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        out = self.act_fn(out)
        return out
