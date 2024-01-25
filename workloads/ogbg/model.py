import torch
from torch_geometric.nn import GIN, MLP, global_add_pool
from ogb.graphproppred import Evaluator
from workloads import WorkloadModel
from runtime.specs import RuntimeSpecs
from submissions import Submission

"""
python submission_runner.py --data_dir $(ws_find bigdata)/data -s adamw_baseline -w ogbg -o $(ws_find bigdata)/experiments/debug --workers 1
"""

class OGBGModel(WorkloadModel):
    """GIN from pytorch geometric"""
    def __init__(
            self,
            submission: Submission,
            node_feature_dim: int,
            num_classes: int,
            dataset_name: str,
            batch_size: int
        ):
        # https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pytorch_lightning/gin.py
        self.batch_size = batch_size

        model = GINwithClassifier(
            node_feature_dim=node_feature_dim,
            num_classes=num_classes
            )
        super().__init__(model, submission)
        self.metric_preds: list[torch.Tensor] = []  # probabilities for class 1
        self.metric_trues: list[torch.Tensor] = []  # labels for classes

        # https://ogb.stanford.edu/docs/home/
        self.evaluator = Evaluator(name=dataset_name)
        # You can learn the input and output format specification of the evaluator as follows.
        # print(self.evaluator.expected_input_format)
            # ==== Expected input format of Evaluator for ogbg-molhiv
            # {'y_true': y_true, 'y_pred': y_pred}
            # - y_true: numpy ndarray or torch tensor of shape (num_graphs, num_tasks)
            # - y_pred: numpy ndarray or torch tensor of shape (num_graphs, num_tasks)
            # where y_pred stores score values (for computing AUC score),
            # num_task is 1, and each row corresponds to one graph.
            # nan values in y_true are ignored during evaluation.
        # print(self.evaluator.expected_output_format)
            # ==== Expected output format of Evaluator for ogbg-molhiv
            # {'rocauc': rocauc}
            # - rocauc (float): ROC-AUC score averaged across 1 task(s)

        # input_dict = {"y_true": y_true, "y_pred": y_pred}
        # result_dict = evaluator.eval(input_dict) # E.g., {"rocauc": 0.7321}

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

    def get_specs(self) -> RuntimeSpecs:
        # TODO have another look at epochs etc
        return RuntimeSpecs(
            max_epochs=50,
            max_steps=51_600,
            devices=1,
            target_metric="val_rocauc",
            target_metric_mode="max"
        )

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
            jumping_knowledge="last"
        ):
        super().__init__()
        self.gin = GIN(
            in_channels = node_feature_dim,
            hidden_channels = hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            jk=jumping_knowledge
        )

        self.classifier = MLP(
            [hidden_channels, hidden_channels, num_classes],
            norm="batch_norm",
            dropout=dropout
        )

    def forward(self, x, edge_index, batch):
        x = self.gin(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.classifier(x)
        return x
