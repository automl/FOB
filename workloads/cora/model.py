import torch
from workloads import WorkloadModel
from runtime.specs import RuntimeSpecs
from submissions import Submission
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class CoraModel(WorkloadModel):
    """simple GCN implementation / GAT from pytorch geometric"""
    def __init__(self, submission: Submission, batch_size: int):
        model_name = "GCN"
        self.batch_size = batch_size
        if model_name == "GCN":
            model = GCN()
        elif model_name == "GAT":
            # TODO pytorch geometric GAT
            model = GAT()
        else:
            NotImplementedError("model not available")
        super().__init__(model, submission)
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

    def get_specs(self) -> RuntimeSpecs:
        # TODO set proper specs
        return RuntimeSpecs(
            max_epochs=100,
            max_steps=None,
            devices=1,
            target_metric="val_acc",
            target_metric_mode="max"
        )

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=32):
        super().__init__()
        self.dropout = 0.5
        # cora dataset:
        num_features = 1433
        num_classes = 7
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
