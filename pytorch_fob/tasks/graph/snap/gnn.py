# implementation taken from https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/

import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

from pytorch_fob.tasks.graph.snap.conv import GNNNode, GNNNodeVirtualnode

class GNN(torch.nn.Module):

    def __init__(
            self,
            num_tasks,
            num_layer = 5,
            emb_dim = 300,
            gnn_type = 'gin',
            virtual_node = True,
            residual = False,
            drop_ratio = 0.5,
            jumping_knowledge = "last",
            graph_pooling = "mean"
        ):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.jumping_knowledge = jumping_knowledge
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNNNodeVirtualnode(
                num_layer,
                emb_dim,
                jumping_knowledge = jumping_knowledge,
                drop_ratio = drop_ratio,
                residual = residual,
                gnn_type = gnn_type
            )
        else:
            self.gnn_node = GNNNode(
                num_layer,
                emb_dim,
                jumping_knowledge = jumping_knowledge,
                drop_ratio = drop_ratio,
                residual = residual,
                gnn_type = gnn_type
            )


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn = torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, 2*emb_dim),
                    torch.nn.BatchNorm1d(2*emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2*emb_dim, 1)
                )
            )
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)
