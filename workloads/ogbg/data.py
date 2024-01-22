# https://github.com/Diego999/pyGAT/blob/master/utils.py

import numpy as np
import scipy.sparse as sp
import torch
from workloads import WorkloadDataModule
from runtime import DatasetArgs
from ogb.graphproppred import PygGraphPropPredDataset
# from torch.utils.data import DataLoader
# from torch_geometric.data import DataLoader
from torch_geometric.loader.dataloader import DataLoader as GeomDataLoader


class OGBGDataModule(WorkloadDataModule):
    """ogbg-molhiv https://ogb.stanford.edu/docs/graphprop/#ogbg-mol
    graph: molecule, nodes: atoms, edges: chemical bonds
    features can be found here https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py 
    """
    def __init__(self, dataset_args: DatasetArgs):
        super().__init__(dataset_args)
        self.data_dir = self.data_dir / "ogbg-molhiv"
        self.batch_size = 128*2
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
        print(f"{dataset.num_node_features=}")
        print(f"{dataset.num_classes=}")
        print(f"{dataset.num_features=}")

    def train_dataloader(self):
        self.check_dataset(self.data_train)
        return GeomDataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        self.check_dataset(self.data_val)
        return GeomDataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        self.check_dataset(self.data_test)
        return GeomDataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.workers, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        self.check_dataset(self.data_predict)
        return GeomDataLoader(self.data_predict, batch_size=self.batch_size, collate_fn=self.collate_fn)


    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """
        dataset = PygGraphPropPredDataset(root=self.data_dir, name=self.dataset_name)
        split_idx = dataset.get_idx_split()
        self.data_train = dataset[split_idx["train"]]
        self.data_val = dataset[split_idx["valid"]]
        self.data_test = dataset[split_idx["test"]]
        if stage == "fit":
            self.data_train = dataset[split_idx["train"]]
        if stage == "validate":
            self.data_val = dataset[split_idx["valid"]]
        if stage == "test":
            self.data_test = dataset[split_idx["test"]]
        if stage == "predict":
            NotImplementedError()
