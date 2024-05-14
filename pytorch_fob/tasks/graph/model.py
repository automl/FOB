import torch
from ogb.graphproppred import Evaluator

from pytorch_fob.tasks import TaskModel
from pytorch_fob.tasks.graph.snap.gnn import GNN
from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.engine.utils import log_warn
from pytorch_fob.optimizers import Optimizer


class OGBGModel(TaskModel):
    """GIN from pytorch geometric"""
    def __init__(
            self,
            optimizer: Optimizer,
            config: TaskConfig,
            dataset_name: str = "ogbg-molhiv",
            num_classes: int = 2
    ):
        # https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pytorch_lightning/gin.py
        self.batch_size = config.batch_size

        gin_params = config.model
        model = GNN(
            num_tasks=num_classes,
            num_layer=gin_params.num_layers,
            emb_dim=gin_params.hidden_channels,
            drop_ratio=gin_params.dropout,
            jumping_knowledge=gin_params.jumping_knowledge,
            virtual_node=gin_params.virtual_node,
            graph_pooling=gin_params.graph_pooling
        )
        super().__init__(model, optimizer, config)
        self.metric_preds: list[torch.Tensor] = []  # probabilities for class 1
        self.metric_trues: list[torch.Tensor] = []  # labels for classes

        # https://ogb.stanford.edu/docs/home/
        # You can learn the input and output format specification of the evaluator as follows.
        # print(self.evaluator.expected_input_format)
        self.evaluator = Evaluator(name=dataset_name)

        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, data) -> torch.Tensor:
        return self.model.forward(data)

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
        all_trues = torch.cat(self.metric_trues).float()
        all_preds = torch.cat(self.metric_preds).float()

        validation_dict = {"y_true": all_trues, "y_pred": all_preds}
        try:
            ogb_score = self.evaluator.eval(validation_dict)
        except ValueError:
            log_warn("Error: Input contains NaN.")
            ogb_score = {"rocauc": 0}
        except RuntimeError:
            log_warn("Error: Cannot compute ROCAUC.")
            ogb_score = {"rocauc": 0}
        self.log(log_label, ogb_score["rocauc"])  # type: ignore

        # free memory
        self.metric_trues.clear()
        self.metric_preds.clear()
