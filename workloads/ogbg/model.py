# from https://github.com/Diego999/pyGAT/blob/master/train.py

import torch
from workloads import WorkloadModel
from runtime.specs import RuntimeSpecs
from submissions import Submission
from gnn import GAT, SpGAT

class OGBGModel(WorkloadModel):
    """GAT https://arxiv.org/pdf/1710.10903.pdf, implementation from https://github.com/Diego999/pyGAT/"""
    def __init__(self, submission: Submission):
        cuda = True
        sparse: bool = False  # GAT with sparse version or not
        nfeat: int = 0  # TODO adapt for dataset
        hidden: int = 8  # Number of hidden units
        nclass: int = 0  # TODO adapt for dataset
        dropout: float = 0.6 # Dropout rate (1 - keep probability)
        nb_heads: int = 8  # Number of head attentions
        alpha: float = 0.2  # Alpha for the leaky_relu

        if sparse:
            model = SpGAT(
                nfeat=nfeat,
                nhid=hidden, 
                nclass=nclass, 
                dropout=dropout, 
                nheads=nb_heads, 
                alpha=alpha)
        else:
            model = GAT(
                nfeat=nfeat, 
                nhid=hidden, 
                nclass=nclass, 
                dropout=dropout, 
                nheads=nb_heads, 
                alpha=alpha)

        # TODO: port to lightning
        """
        # Load data
        adj, features, labels, idx_train, idx_val, idx_test = load_data()
        if cuda:
            model.cuda()
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()
        """
        super().__init__(model, submission)
        loss = torch.nn.CrossEntropyLoss
        # datasets:
        #   ogbg-molhiv
        #   https://ogb.stanford.edu/docs/graphprop/
        # task:
        #   graph property prediction
        # metric:
        #   ROC-AUC
        # self.loss_fn = None # TODO
        self.loss_fn = torch.nn.functional.nll_loss



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        # optimizer.zero_grad()  # Clear gradients.
        
        # TODO batch depends on how we load data
        # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
        # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        # labels = encode_onehot(idx_features_labels[:, -1])
        # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
        # build symmetric adjacency matrix
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize_features(features)
        # adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        
        output = self.model(features, adj)
        loss_train = self.loss_fn(output[idx_train], labels[idx_train])
        self.loss_fn.backward()  # Derive gradients.
        # optimizer.step()  # Update parameters based on gradients.
        return loss, h

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def get_specs(self) -> RuntimeSpecs:
        raise NotImplementedError
