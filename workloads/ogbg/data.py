# https://github.com/Diego999/pyGAT/blob/master/utils.py

import numpy as np
import scipy.sparse as sp
import torch.utils.data as torch_data
from workloads import WorkloadDataModule
from runtime import DatasetArgs
from ogb.graphproppred import PygGraphPropPredDataset
import torch_geometric.data as geom_data

class OGBGDataModule(WorkloadDataModule):
    """ogbg-molhiv https://ogb.stanford.edu/docs/graphprop/#ogbg-mol
    graph: molecule, nodes: atoms, edges: chemical bonds
    features can be found here https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py 
    """
    def __init__(self, dataset_args: DatasetArgs):
        super().__init__(dataset_args)
        self.data_dir = self.data_dir / "ogbg-molhiv"
        self.batch_size = 32
        # ogbg-molhiv is small (41,127 graphs)
        # ogbg-molpcba is medium size (437,929 graphs)
        self.dataset_name = "ogbg-molhiv"
        # self.dataset_name = "ogbg-molpcba"
        
        # TODO, checkl if we need those, get them from model
        dataset = PygGraphPropPredDataset(root=self.data_dir, name=self.dataset_name)
        print(f"{dataset.num_node_features=}")
        print(f"{dataset.num_classes=}")
        print(f"{dataset.num_features=}")
        self.feature_dim: int = dataset.num_features
        self.num_classes: int = dataset.num_classes
        
    def prepare_data(self):
        """Load citation network dataset (cora only for now)"""
        dataset = PygGraphPropPredDataset(root=self.data_dir, name=self.dataset_name) 
        split_idx = dataset.get_idx_split()
        self.data_train = dataset[split_idx["train"]]
        self.data_val = dataset[split_idx["valid"]]
        self.data_test = dataset[split_idx["test"]]
        print(f"{dataset.num_node_features=}")
        print(f"{dataset.num_classes=}")
        print(f"{dataset.num_features=}")

    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """
        dataset = PygGraphPropPredDataset(root=self.data_dir, name=self.dataset_name) 
        split_idx = dataset.get_idx_split()
        self.data_train = dataset[split_idx["train"]]
        self.data_val = dataset[split_idx["valid"]]
        self.data_test = dataset[split_idx["test"]]
        if stage == "fit":
            return torch_data.DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.workers)
        if stage == "validate":
            return geom_data.DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.workers)
        if stage == "test":
            return geom_data.DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.workers)
        if stage == "predict":
            return geom_data.DataLoader(self.data_predict, batch_size=self.batch_size, num_workers=self.workers)
