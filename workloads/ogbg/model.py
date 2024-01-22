import torch
from workloads import WorkloadModel
from runtime.specs import RuntimeSpecs
from submissions import Submission
from torch_geometric.nn import GAT
from torchmetrics import Accuracy
from torch_geometric.nn import GAT, GIN, MLP, global_add_pool
from ogb.graphproppred import Evaluator

"""
python submission_runner.py --data_dir $(ws_find bigdata)/data -s adamw_baseline -w ogbg -o $(ws_find bigdata)/experiments --workers 1
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

        self.loss_fn = torch.nn.CrossEntropyLoss()


    def forward(self, x, edge_index, batch) -> torch.Tensor:
        x = self.model.forward(x, edge_index, batch)
        # TODO maybe bring back later
        # x = global_add_pool(x, batch)
        # x = self.classifier(x)
        return x

    def training_step(self, data, batch_idx):
        y_hat = self.forward(data.x, data.edge_index, data.batch)
        # import pdb; pdb.set_trace()
        loss = self.loss_fn(y_hat, data.y.squeeze(dim=-1))
        # self.train_acc(y_hat.softmax(dim=-1), data.y)
        #self.train_acc(y_hat.softmax(dim=-1), data.y.squeeze(dim=-1))
        #self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        
        # TODO read batch_size from data instead?
        self.log("loss", loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, data, batch_idx):
        y_hat = self.forward(data.x, data.edge_index, data.batch)
        # import pdb; pdb.set_trace()
        # TODO: use softmax fpr loss?
        loss = self.loss_fn(y_hat, data.y.squeeze(dim=-1))
        # import pdb; pdb.set_trace()
        # y_hat.shape:                                      torch.Size([32, 2])
        # data.y.shape:                                     torch.Size([32, 1])
        # y_hat.argmax(dim=-1).unsqueeze(dim=-1).shape:     torch.Size([32, 1])
        try:
            ogb_score = self.evaluator.eval({"y_true": data.y, "y_pred": y_hat.argmax(dim=-1).unsqueeze(dim=-1)})
            self.log("val_rocauc", ogb_score["rocauc"], batch_size=self.batch_size)
        except RuntimeError:  # No positively labeled data available. Cannot compute ROC-AUC.
            ogb_score = {"rocauc": 0}  # TODO: what to do here? righ now we simply do not log it
            print("\nexception caught\n")
        
        self.log("val_loss", loss, batch_size=self.batch_size)
        # self.val_acc(y_hat.softmax(dim=-1), data.y)
        #self.val_acc(y_hat.softmax(dim=-1), data.y.squeeze(dim=-1))
        #self.log('val_acc', self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, data, batch_idx):
        y_hat = self.forward(data.x, data.edge_index, data.batch)
        # self.test_acc(y_hat.softmax(dim=-1), data.y)
        #self.test_acc(y_hat.softmax(dim=-1), data.y.squeeze(dim=-1))
        #self.log('test_acc', self.test_acc, prog_bar=True, on_step=False, on_epoch=True)
        loss = self.loss_fn(y_hat, data.y.squeeze(dim=-1))
        try:
            ogb_score = self.evaluator.eval({"y_true": data.y, "y_pred": y_hat.argmax(dim=-1).unsqueeze(dim=-1)})
        except RuntimeError:  # No positively labeled data available. Cannot compute ROC-AUC.
            ogb_score = {"rocauc": 0}  # TODO: what to do here?
            print("\nexception caught\n")
        
        self.log("test_loss", loss, batch_size=self.batch_size)
        self.log("test_rocauc", ogb_score["rocauc"], batch_size=self.batch_size)

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
    def __init__(self, node_feature_dim, num_classes, hidden_channels=64, num_layers=5, dropout=0.3, jumping_knowledge="cat"):
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
