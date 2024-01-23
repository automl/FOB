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
        self.validation_step_outputs: list[torch.tensor] = []
        self.test_step_outputs: list[torch.tensor] = []

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

        # TODO: use softmax for loss?
        loss = self.loss_fn(y_hat, data.y.squeeze(dim=-1))
        self.log("val_loss", loss, batch_size=self.batch_size)
        
        # y_hat.shape:                                      torch.Size([batch_size, 2])
        # data.y.shape:                                     torch.Size([batch_size, 1])
        # y_hat.argmax(dim=-1).unsqueeze(dim=-1).shape:     torch.Size([batch_size, 1])
        
        # Validation of single step in theory, but this throws an exception if there is no positive label
        # step_dict = {"y_true": data.y, "y_pred": y_hat.argmax(dim=-1).unsqueeze(dim=-1)}
        # return step_dict
        # shape torch.Size([2, batch_size, 1])
        pred = torch.stack((data.y, y_hat.argmax(dim=-1).unsqueeze(dim=-1)))
        self.validation_step_outputs.append(pred)
        
        # TODO: what do we want to return? loss? dict with loss and pred?
        return 
    
    def on_validation_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-epoch-level-metrics
        """
        # self.validation_step_outputs is a list of whatever we set in *validation_step*
        almost_all_preds = torch.stack(self.validation_step_outputs[:-1])  # torch.Size([n_eval_steps, 2, batch_size, 1])
        last_pred = self.validation_step_outputs[-1]
        last_y, last_y_hat = last_pred.split(1, dim=0)
        last_y = last_y.squeeze(dim=0)
        last_y_hat = last_y_hat.squeeze(dim=0)

        # build aggregated dictionary with all results
        eval_steps, _, batch_size, _ = almost_all_preds.size()
        reshaped_preds = almost_all_preds.view(2, eval_steps * batch_size, 1)
        ys: torch.tensor
        y_hats: torch.tensor
        ys, y_hats = reshaped_preds.split(1, dim=0)
        # todo, is this performant?
        ys = ys.squeeze(dim=0)
        y_hats = y_hats.squeeze(dim=0)

        ys = torch.cat((ys, last_y))
        y_hats = torch.cat((y_hats, last_y_hat))
        
        validation_dict = {"y_true": ys, "y_pred": y_hats}

        ogb_score = self.evaluator.eval(validation_dict)
        # self.log("val_rocauc", ogb_score["rocauc"], batch_size=self.batch_size)
        self.log("val_rocauc", ogb_score["rocauc"])
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, data, batch_idx):
        y_hat = self.forward(data.x, data.edge_index, data.batch)
        # self.test_acc(y_hat.softmax(dim=-1), data.y)
        # self.test_acc(y_hat.softmax(dim=-1), data.y.squeeze(dim=-1))
        # self.log('test_acc', self.test_acc, prog_bar=True, on_step=False, on_epoch=True)
        loss = self.loss_fn(y_hat, data.y.squeeze(dim=-1))

        res = torch.stack((data.y, y_hat.argmax(dim=-1).unsqueeze(dim=-1)))
        self.test_step_outputs.append(res)

        # TODO: what do we want to return? loss? dict with loss and pred?
        return
    
        try:
            ogb_score = self.evaluator.eval({"y_true": data.y, "y_pred": y_hat.argmax(dim=-1).unsqueeze(dim=-1)})
        except RuntimeError:  # No positively labeled data available. Cannot compute ROC-AUC.
            ogb_score = {"rocauc": 0}  # TODO: what to do here?
            print("\nexception caught\n")
        
        self.log("test_loss", loss, batch_size=self.batch_size)
        self.log("test_rocauc", ogb_score["rocauc"], batch_size=self.batch_size)

    def on_test_epoch_end(self) -> None:
        almost_all_preds = torch.stack(self.test_step_outputs[:-1])  # torch.Size([n_eval_steps, 2, batch_size, 1])
        last_pred = self.test_step_outputs[-1]
        last_y, last_y_hat = last_pred.split(1, dim=0)
        last_y = last_y.squeeze(dim=0)
        last_y_hat = last_y_hat.squeeze(dim=0)

        # build aggregated dictionary with all results
        eval_steps, _, batch_size, _ = almost_all_preds.size()
        reshaped_preds = almost_all_preds.view(2, eval_steps * batch_size, 1)
        ys: torch.tensor
        y_hats: torch.tensor
        ys, y_hats = reshaped_preds.split(1, dim=0)
        # todo, is this performant?
        ys = ys.squeeze(dim=0)
        y_hats = y_hats.squeeze(dim=0)

        ys = torch.cat((ys, last_y))
        y_hats = torch.cat((y_hats, last_y_hat))

        test_dict = {"y_true": ys, "y_pred": y_hats}

        ogb_score = self.evaluator.eval(test_dict)
        self.log("test_rocauc", ogb_score["rocauc"])

        self.test_step_outputs.clear()  # free memory

    def get_specs(self) -> RuntimeSpecs:
        # TODO have another look at epochs etc
        return RuntimeSpecs(
            max_epochs=100,
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
