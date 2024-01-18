import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch.utils.data import DataLoader
from workloads import WorkloadDataModule
from runtime import DatasetArgs

class CoraDataModule(WorkloadDataModule):
    """https://colab.research.google.com/drive/14OvFnAXggxB8vM4e8vSURUp1TaKnovzX?usp=sharing#scrollTo=imGrKO5YH11-"""
    def __init__(self, dataset_args: DatasetArgs):
        super().__init__(dataset_args)
        
    def prepare_data(self):
        """Load citation network dataset (cora only for now)"""
        self.data_dir = self.data_dir
        dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
        print(f'Dataset: {dataset}:')
        print('======================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')



    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """
        if stage == "fit":
            self.data_train = ...
        if stage == "validate":
            self.data_val = ...
        if stage == "test":
            self.data_test = ...
        if stage == "predict":
            self.data_predict = ...
