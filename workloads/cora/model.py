import torch
from workloads import WorkloadModel
from runtime.specs import RuntimeSpecs
from submissions import Submission
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class CoraModel(WorkloadModel):
    """GAT from pytorch geometric"""
    def __init__(self, submission: Submission):
        model = GCN()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        super().__init__(model, submission)


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
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log("test_acc", acc)

    def get_specs(self) -> RuntimeSpecs:
        raise NotImplementedError

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=16):
        super().__init__()
        # torch.manual_seed(1234567)
        
        # cora dataset:
        #   Number of features: 1433
        #   Number of classes: 7
        num_features = 1433
        num_classes = 7
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x