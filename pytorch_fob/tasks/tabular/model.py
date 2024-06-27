import torch
from sklearn.metrics import r2_score, root_mean_squared_error
from rtdl_revisiting_models import FTTransformer, _CLSEmbedding, LinearEmbeddings, CategoricalEmbeddings
from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.engine.parameter_groups import GroupedModel, ParameterGroup, group_named_parameters
from pytorch_fob.tasks import TaskModel
from pytorch_fob.optimizers import Optimizer


class GroupedFTTransformer(GroupedModel):
    def parameter_groups(self) -> list[ParameterGroup]:
        blacklist_modules = (_CLSEmbedding, LinearEmbeddings, CategoricalEmbeddings)
        # same conditions as in FTTransformer.make_parameter_groups()
        apply_no_decay_conds = [
            lambda m, _, pn: isinstance(m, blacklist_modules),
            lambda m, _, pn: pn.endswith('bias'),
            lambda m, _, pn: pn.endswith('_normalization'),
        ]
        # ignore top-level (params are added through its children):
        ignore_conds = [lambda m, p, pn: isinstance(m, FTTransformer)]
        return group_named_parameters(
            self.model,
            g1_conds=apply_no_decay_conds,
            g1_kwargs={'weight_decay_multiplier': 0.0},
            ignore_conds=ignore_conds
        )


class TabularModel(TaskModel):
    """
    Lightning Module for tabular data task.
    Model is FT-Transformer from https://arxiv.org/abs/2106.11959v5.
    """
    def __init__(self, optimizer: Optimizer, config: TaskConfig):
        # output dimension
        d_out = 1
        # Continuous features. (depends on dataset)
        n_cont_features = 8
        # Categorical features. (depends on dataset)
        cat_cardinalities = []
        n_blocks = config.model.n_blocks
        default_kwargs = FTTransformer.get_default_kwargs(n_blocks)
        model = FTTransformer(
            n_cont_features=n_cont_features,
            cat_cardinalities=cat_cardinalities,
            d_out=d_out,
            **default_kwargs,
        )
        super().__init__(GroupedFTTransformer(model), optimizer, config)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x_cont=x, x_cat=None).squeeze(-1)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        features, labels = batch
        preds = self.forward(features)
        loss = self.compute_and_log_loss(preds, labels, "train_loss")
        self.compute_and_log_metrics(preds, labels, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        preds = self.forward(features)
        self.compute_and_log_loss(preds, labels, "val_loss")
        self.compute_and_log_metrics(preds, labels, stage="val")

    def test_step(self, batch, batch_idx):
        features, labels = batch
        preds = self.forward(features)
        self.compute_and_log_loss(preds, labels, "test_loss")
        self.compute_and_log_metrics(preds, labels, stage="test")

    def compute_and_log_loss(self, preds: torch.Tensor, targets: torch.Tensor, log_label: str) -> torch.Tensor:
        loss = self.loss_fn(preds, targets)
        self.log(log_label, loss)
        return loss

    def compute_and_log_metrics(self, preds: torch.Tensor, targets: torch.Tensor, stage: str):
        preds = preds.detach().cpu().float().numpy()
        targets = targets.detach().cpu().float().numpy()
        metrics = {
            "rmse": root_mean_squared_error(targets, preds),
            "r2_score": r2_score(targets, preds)
        }
        for k, v in metrics.items():
            self.log(f"{stage}_{k}", v) # type: ignore
        return metrics
