import torch
from tasks import TaskModel
from engine.configs import TaskConfig
from optimizers import Optimizer
from torch_geometric.nn import GCNConv
from torch import nn
import torch.nn.functional as F
from tasks.graph_tiny.gcn import GCN as OurGCN


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
        model_name = config.model.name
        if model_name == "GCN":
            model = GCN(hidden_channels=hidden_channels,
                    num_layers=num_layers,
                    dropout=dropout,
                    cached=cached,
                    normalize=normalize,
                    reset_params=reset_params)
        elif model_name == "our_GCN":
            if num_layers != 2:
                # also ignoring cached, normalize, reset_params from config atm
                raise NotImplementedError()
            nfeat = 1433
            nclass = 7
            model = OurGCN(nfeat=nfeat,
                           nhid=hidden_channels,
                           nclass=nclass,
                           dropout=dropout)

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
                    torch.nn.init.uniform_(param, -stdv, stdv)
                else:  # bias
                    stdv = 1. / torch.sqrt(torch.tensor(param.size(0)))
                    torch.nn.init.uniform_(param, -stdv.item(), stdv.item())

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
