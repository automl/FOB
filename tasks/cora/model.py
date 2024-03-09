import torch
from tasks import TaskModel
from engine.configs import TaskConfig
from optimizers import Optimizer
from torch_geometric.nn import GCNConv
from torch import nn
import torch.nn.functional as F


class CoraModel(TaskModel):
    """simple GCN implementation / GAT from pytorch geometric"""
    def __init__(self, optimizer: Optimizer, config: TaskConfig):
        self.batch_size = config.batch_size
        hidden_channels = config.model.hidden_channels
        num_layers = config.model.num_layers
        cached = config.model.cached
        normalize = config.model.normalize
        dropout = config.model.dropout
        model = GCN(hidden_channels=hidden_channels,
                    num_layers=num_layers,
                    dropout=dropout,
                    cached=cached,
                    normalize=normalize)
        super().__init__(model, optimizer, config)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, data: torch.Tensor, mode="train") -> torch.Tensor:
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
            assert False, "Unknown forward mode: %s" % mode

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
    def __init__(self, hidden_channels=32, num_layers:int = 2, dropout=0.5, cached:bool=False, normalize:bool=True):
        self.dropout = dropout
        self.num_layers = num_layers
        super().__init__()
        # cora dataset:
        num_features = 1433
        num_classes = 7

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


    def forward(self, x, edge_index):
        # print(edge_index)
        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if idx < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = F.log_softmax(x, dim=1)

        return x
