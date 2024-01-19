# https://github.com/Diego999/pyGAT/blob/master/utils.py

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader
from workloads import WorkloadDataModule
from runtime import DatasetArgs
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader

class OGBGDataModule(WorkloadDataModule):
    """ogbg-molhiv https://ogb.stanford.edu/docs/graphprop/#ogbg-mol
    graph: molecule, nodes: atoms, edges: chemical bonds
    features can be found here https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py 
    """
    def __init__(self, dataset_args: DatasetArgs):
        super().__init__(dataset_args)
        self.data_dir = self.data_dir / "ogbg-molhiv"
        self.batch_size = 32
        self.dataset_name = "ogbg-molhiv"
        
        # TODO, checkl if we need those, get them from model
        self.feature_dim = ...
        self.num_classes = ...
        
    def prepare_data(self):
        """Load citation network dataset (cora only for now)"""
        dataset = PygGraphPropPredDataset(root=self.data_dir, name=self.dataset_name, ) 
        split_idx = dataset.get_idx_split()
        self.data_train = dataset[split_idx["train"]]
        self.data_val = dataset[split_idx["valid"]]
        self.data_test = dataset[split_idx["test"]]

    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """
        if stage == "fit":
            return self.train_dataloader()
        if stage == "test":
            return self.val_dataloader()
        if stage == "predict":
            return self.test_dataloader()
    