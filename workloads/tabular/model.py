import torch
from rtdl_revisiting_models import FTTransformer
from workloads import WorkloadModel
from runtime.configs import WorkloadConfig
from submissions import Submission


class TabularModel(WorkloadModel):
    """
    Lightning Module for tabular data task.
    Model is FT-Transformer from https://arxiv.org/abs/2106.11959v5.
    """
    def __init__(self, submission: Submission, workload_config: WorkloadConfig):
        # output dimension
        d_out = 1
        # Continuous features. (depends on dataset)
        n_cont_features = 8
        # Categorical features. (depends on dataset)
        cat_cardinalities = []
        n_blocks = workload_config.model.n_blocks
        default_kwargs = FTTransformer.get_default_kwargs(n_blocks)
        model = FTTransformer(
            n_cont_features=n_cont_features,
            cat_cardinalities=cat_cardinalities,
            d_out=d_out,
            **default_kwargs,
        )
        super().__init__(model, submission, workload_config)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x_cont=x, x_cat=None).squeeze(-1)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        features, labels = batch
        preds = self.forward(features)
        loss = self.compute_and_log_loss(preds, labels, "train_loss")
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        preds = self.forward(features)
        self.compute_and_log_loss(preds, labels, "val_loss")

    def test_step(self, batch, batch_idx):
        features, labels = batch
        preds = self.forward(features)
        self.compute_and_log_loss(preds, labels, "test_loss")

    def compute_and_log_loss(self, preds: torch.Tensor, targets: torch.Tensor, log_label: str) -> torch.Tensor:
        loss = self.loss_fn(preds, targets)
        self.log(log_label, loss)
        return loss
