# https://github.com/Diego999/pyGAT/blob/master/utils.py

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader
from workloads import WorkloadDataModule
from runtime import DatasetArgs

class OGBGDataModule(WorkloadDataModule):
    def __init__(self, dataset_args: DatasetArgs):
        super().__init__(dataset_args)
        self.batch_size = 1  # TODO
        self.train_val_split = [1, 1]  # TODO
        self.seed = 42

        # TODO do we need to normalize?
        # meanOfOpenWebText = torch.tensor(0)
        # stdOfOpenWebText = torch.tensor(1)
        # self.transform  = transforms.Compose([transforms.ToTensor(),transforms.Normalize(meanOfOpenWebText, stdOfOpenWebText)])  
        
    def prepare_data(self):
        """Load citation network dataset (cora only for now)"""
        dataset = "cora"
        self.data_dir = self.data_dir
        print(f"Loading {dataset} dataset...")
        if dataset == "cora":
            labels_file =  self.data_dir / "cora.content"
            edges_file =  self.data_dir / "cora.cites"

            idx_features_labels = np.genfromtxt(labels_file, dtype=np.dtype(str))
            features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
            labels = encode_onehot(idx_features_labels[:, -1])

            # build graph
            idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
            idx_map = {j: i for i, j in enumerate(idx)}
            edges_unordered = np.genfromtxt(edges_file, dtype=np.int32)
            edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

            # build symmetric adjacency matrix
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

            features = normalize_features(features)
            adj = normalize_adj(adj + sp.eye(adj.shape[0]))

            split_train = range(140)
            split_val = range(200, 500)
            split_test = range(500, 1500)

            adj = torch.FloatTensor(np.array(adj.todense()))
            features = torch.FloatTensor(np.array(features.todense()))
            labels = torch.LongTensor(np.where(labels)[1])

            idx_train = torch.LongTensor(idx_train)
            idx_val = torch.LongTensor(idx_val)
            idx_test = torch.LongTensor(idx_test)


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
    



def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)