import torch
from torch_geometric.nn import GIN, MLP, global_add_pool
from ogb.graphproppred import Evaluator
from workloads import WorkloadModel
from engine.configs import WorkloadConfig
from submissions import Submission


class OGBGModel(WorkloadModel):
    """GIN from pytorch geometric"""
    def __init__(
            self,
            submission: Submission,
            node_feature_dim: int,
            num_classes: int,
            dataset_name: str,
            batch_size: int,
            workload_config: WorkloadConfig
    ):
        # https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pytorch_lightning/gin.py
        self.batch_size = batch_size

        gin_params = workload_config.model
        mlp_params = workload_config.model.mlp
        model = GINwithClassifier(
            node_feature_dim=node_feature_dim,
            num_classes=num_classes,
            hidden_channels=gin_params.hidden_channels,
            num_layers=gin_params.num_layers,
            dropout=gin_params.dropout,
            jumping_knowledge=gin_params.jumping_knowledge,
            classifier_hidden_channel=mlp_params.hidden_channels,
            classifier_num_layers=mlp_params.layers,
            classifier_dropout=mlp_params.dropout,
            classifier_norm=mlp_params.norm
            )
        super().__init__(model, submission, workload_config)
        self.metric_preds: list[torch.Tensor] = []  # probabilities for class 1
        self.metric_trues: list[torch.Tensor] = []  # labels for classes

        # https://ogb.stanford.edu/docs/home/
        # You can learn the input and output format specification of the evaluator as follows.
        # print(self.evaluator.expected_input_format)
        self.evaluator = Evaluator(name=dataset_name)

        self.loss_fn = torch.nn.BCELoss()

    def forward(self, data) -> torch.Tensor:
        return self.model.forward(data.x, data.edge_index, data.batch).softmax(dim=-1)

    def training_step(self, data, batch_idx):
        y_hat = self.forward(data)
        return self.compute_and_log_loss(y_hat, data.y, "train_loss")

    def validation_step(self, data, batch_idx):
        y_hat = self.forward(data)
        self.compute_and_log_loss(y_hat, data.y, "val_loss")
        self._collect_data_for_metric(y_hat, data.y)

    def test_step(self, data, batch_idx):
        y_hat = self.forward(data)
        self._collect_data_for_metric(y_hat, data.y)

    def compute_and_log_loss(self, preds, labels, log_label: str):
        labels = labels.to(torch.float32).squeeze(dim=-1)  # floats for BCE but int for rocauc
        loss = self.loss_fn(preds[:, 1], labels)
        self.log(log_label, loss, batch_size=self.batch_size)
        return loss

    def _collect_data_for_metric(self, preds, labels):
        self.metric_preds.append(preds[:, 1].unsqueeze(dim=-1))
        self.metric_trues.append(labels)

    def on_validation_epoch_end(self):
        self.compute_and_log_metric("val_rocauc")

    def on_test_epoch_end(self) -> None:
        self.compute_and_log_metric("test_rocauc")

    def compute_and_log_metric(self, log_label: str):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-epoch-level-metrics
        """
        all_trues = torch.cat(self.metric_trues)
        all_preds = torch.cat(self.metric_preds)

        validation_dict = {"y_true": all_trues, "y_pred": all_preds}
        ogb_score = self.evaluator.eval(validation_dict)
        self.log(log_label, ogb_score["rocauc"])  # type: ignore

        # free memory
        self.metric_trues.clear()
        self.metric_preds.clear()


class GINwithClassifier(torch.nn.Module):
    def __init__(
        self,
        node_feature_dim,
        num_classes,
        hidden_channels=300,
        num_layers=5,
        dropout=0.5,
        jumping_knowledge="last",
        classifier_hidden_channel=300,
        classifier_num_layers=2,
        classifier_norm="batch_norm",
        classifier_dropout=0.5
    ):
        super().__init__()
        self.gin = GIN(
            in_channels=node_feature_dim,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            jk=jumping_knowledge
        )

        self.classifier = MLP(
            in_channels=hidden_channels,
            hidden_channels=classifier_hidden_channel,
            out_channels=num_classes,
            num_layers=classifier_num_layers,
            norm=classifier_norm,
            dropout=classifier_dropout
        )

    def forward(self, x, edge_index, batch):
        x = self.gin(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.classifier(x)
        return x
