import torch
from workloads import WorkloadModel
from runtime.specs import RuntimeSpecs
from submissions import Submission
from torch_geometric.nn import GCNConv


class CoraModel(WorkloadModel):
    """GAT from pytorch geometric"""
    def __init__(self, submission: Submission):
        model = ...
        super().__init__(model, submission)
        self.loss_fn = ...



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def get_specs(self) -> RuntimeSpecs:
        raise NotImplementedError

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # torch.manual_seed(1234567)
        
        # cora dataset:
        #   Number of features: 1433
        #   Number of classes: 7
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x