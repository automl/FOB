import torch
from workloads import WorkloadModel
from runtime.specs import RuntimeSpecs
from submissions import Submission
from torch_geometric.nn import GAT
from torchmetrics import Accuracy
from torch_geometric.nn import GAT, GIN, MLP, global_add_pool
from ogb.graphproppred import Evaluator

"""
python submission_runner.py --data_dir $(ws_find bigdata)/data -s adamw_baseline -w ogbg -o $(ws_find bigdata)/experiments/debug --workers 1
"""

class OGBGModel(WorkloadModel):
    """GIN from pytorch geometric"""
    def __init__(self, submission: Submission, node_feature_dim: int, num_classes: int, dataset_name: str, batch_size: int):
        # https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pytorch_lightning/gin.py
        self.batch_size = batch_size

        model = GINwithClassifier(
            node_feature_dim=node_feature_dim,
            num_classes=num_classes
            )
        super().__init__(model, submission)
        self.validation_step_trues: list[torch.tensor] = []
        self.validation_step_preds: list[torch.tensor] = []
        self.test_step_trues: list[torch.tensor] = []
        self.test_step_preds: list[torch.tensor] = []

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


    def forward(self, x, edge_index, batch) -> torch.Tensor:
        x = self.model.forward(x, edge_index, batch)
        return x

    def training_step(self, data, batch_idx):
        y_hat = self.forward(data.x, data.edge_index, data.batch)
        mask = data.y.squeeze(dim=-1)
        prediction_for_correct_class = y_hat.softmax(dim=-1)[torch.arange(y_hat.size(0)), mask].unsqueeze(dim=-1)
        labels = data.y.to(torch.float32)  # floats for BCE but int for rocauc
        loss = self.loss_fn(prediction_for_correct_class, labels)
        # TODO read batch_size from data instead?
        self.log("loss", loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, data, batch_idx):
        y_hat = self.forward(data.x, data.edge_index, data.batch)
        # import pdb; pdb.set_trace()
        
        # y_hat.shape:                                      torch.Size([batch_size, 2])
        # data.y.shape:                                     torch.Size([batch_size, 1])
        mask = data.y.squeeze(dim=-1)
        prediction_for_correct_class = y_hat.softmax(dim=-1)[torch.arange(y_hat.size(0)), mask].unsqueeze(dim=-1)
        labels = data.y.to(torch.float32)  # floats for BCE but int for rocauc
        # import pdb; pdb.set_trace()
        loss = self.loss_fn(prediction_for_correct_class, labels)
        self.log("val_loss", loss, batch_size=self.batch_size)

        # those are aggregated after all steps
        self.validation_step_preds.append(prediction_for_correct_class)
        self.validation_step_trues.append(data.y)
        
        # we do not test single epochs, as not having positive labels
        # (which will happen in small batches makes it impossible to calculate the rocauc)
        # dic = {"y_true": data.y, "y_pred": prediction_for_correct_class}
        # import pdb; pdb.set_trace()
        # self.evaluator.eval(dic)
        return
    
    def on_validation_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-epoch-level-metrics
        """
        # import pdb; pdb.set_trace()

        all_trues = torch.cat(self.validation_step_trues)
        all_preds = torch.cat(self.validation_step_preds)

        validation_dict = {"y_true": all_trues, "y_pred": all_preds}

        ogb_score = self.evaluator.eval(validation_dict)
        self.log("val_rocauc", ogb_score["rocauc"])
        
        # free memory
        self.validation_step_trues.clear()
        self.validation_step_preds.clear()

    def test_step(self, data, batch_idx):
        y_hat = self.forward(data.x, data.edge_index, data.batch)        
        mask = data.y.squeeze(dim=-1)
        prediction_for_correct_class = y_hat.softmax(dim=-1)[torch.arange(y_hat.size(0)), mask].unsqueeze(dim=-1)
        # aggregate over this data in on_test_epoch_end
        self.test_step_preds.append(prediction_for_correct_class)
        self.test_step_trues.append(data.y)


    def on_test_epoch_end(self) -> None:
        """aggregate all the test steps
        """
        all_trues = torch.cat(self.test_step_trues)
        all_preds = torch.cat(self.test_step_preds)

        test_dict = {"y_true": all_trues, "y_pred": all_preds}

        ogb_score = self.evaluator.eval(test_dict)
        self.log("test_rocauc", ogb_score["rocauc"])

        # free memory
        self.test_step_trues.clear()
        self.test_step_preds.clear()

    def get_specs(self) -> RuntimeSpecs:
        # TODO have another look at epochs etc
        return RuntimeSpecs(
            max_epochs=50,
            max_steps=None,
            devices=1,
            target_metric="val_rocauc",
            target_metric_mode="max"
        )

class GINwithClassifier(torch.nn.Module):
    def __init__(self, node_feature_dim, num_classes, hidden_channels=300, num_layers=5, dropout=0.5, jumping_knowledge="last"):
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
