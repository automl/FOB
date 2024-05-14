import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch_geometric.nn import GCNConv
import torch_geometric.data as geom_data
from pytorch_fob.tasks import TaskModel
from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.optimizers import Optimizer


class CoraModel(TaskModel):
    """simple GCN implementation from pytorch geometric"""
    def __init__(self, optimizer: Optimizer, config: TaskConfig):
        self.batch_size = config.batch_size
        hidden_channels = config.model.hidden_channels
        num_layers = config.model.num_layers
        cached = config.model.cached
        normalize = config.model.normalize
        dropout = config.model.dropout
        reset_params = config.model.reset_params
        model = GCN(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            cached=cached,
            normalize=normalize,
            reset_params=reset_params
        )

        super().__init__(model, optimizer, config)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, data: geom_data.data.BaseData, mode="train") -> tuple[Tensor, Tensor]:
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)

        # Only calculate the loss on the nodes corresponding to the mask
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, f"Unknown forward mode: {mode}"

        loss = self.loss_fn(x[mask], data.y[mask])
        acc = (x[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log("train_loss", loss, on_epoch=True, batch_size=self.batch_size)
        self.log("train_acc", acc, on_epoch=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log("val_acc", acc, on_epoch=True, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log("test_acc", acc, on_epoch=True, batch_size=self.batch_size)


class GCN(torch.nn.Module):
    def __init__(
            self,
            num_features=1433,
            num_classes=7,
            hidden_channels=32,
            num_layers: int = 2,
            dropout=0.5,
            cached: bool = False,
            normalize: bool = True,
            reset_params: bool = False
            ):
        self.dropout = dropout
        self.num_layers = num_layers
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(num_features,
                    hidden_channels,
                    cached=cached,
                    normalize=normalize,
                    dropout=dropout
                    )
        )

        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(
                    hidden_channels,
                    hidden_channels,
                    cached=cached,
                    normalize=normalize,
                    dropout=dropout
                )
            )

        self.convs.append(
            GCNConv(
                hidden_channels,
                num_classes,
                cached=cached,
                normalize=normalize,
            )
        )
        if reset_params:
            self._reset_parameters()

    def _reset_parameters(self):
        """
        initialization from https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
        which follows https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        """
        for conv in self.convs:
            for param in conv.parameters():
                if param.dim() > 1:  # weight parameter
                    stdv = 1. / torch.sqrt(torch.tensor(param.size(1)))
                    nn.init.uniform_(param, -stdv.item(), stdv.item())
                else:  # bias
                    stdv = 1. / torch.sqrt(torch.tensor(param.size(0)))
                    nn.init.uniform_(param, -stdv.item(), stdv.item())

    def forward(self, x, edge_index):
        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            not_last: bool = idx < self.num_layers - 1
            if not_last:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = F.log_softmax(x, dim=1)

        return x
