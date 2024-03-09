import torch_geometric.loader as geom_loader
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from tasks import TaskDataModule
from engine.configs import TaskConfig


class CoraDataModule(TaskDataModule):
    """https://colab.research.google.com/drive/14OvFnAXggxB8vM4e8vSURUp1TaKnovzX?usp=sharing#scrollTo=imGrKO5YH11-
    https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html?highlight=planetoid#common-benchmark-datasets
    https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/06-graph-neural-networks.html"""
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.batch_size = 1  # As we have a single graph, we use a batch size of 1
        self.data_dir = self.data_dir / "Planetoid"
        self.split = config.dataset_split

    def prepare_data(self):
        """Load citation network dataset (cora)"""
        self.data_dir.mkdir(exist_ok=True)

        # dataset split:
        # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html
        # The type of dataset split ("public", "full", "geom-gcn", "random").
        # If set to "public",
        #   the split will be the public fixed split from the
        #   “Revisiting Semi-Supervised Learning with Graph Embeddings” paper.
        # If set to "full",
        #   all nodes except those in the validation and test sets will be used for training
        #   (as in the “FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling” paper).
        # If set to "geom-gcn", the 10 public fixed splits from the
        #   “Geom-GCN: Geometric Graph Convolutional Networks” paper are given.
        # If set to "random",
        #   train, validation, and test sets will be randomly generated,
        #   according to num_train_per_class, num_val and num_test. (default: "public")

        dataset = Planetoid(root=self.data_dir, name='Cora', split=self.split, transform=NormalizeFeatures())

        print_cora_stats = True
        if print_cora_stats:
            print()
            print(f'Dataset: {dataset}:')
            print('======================')
            print(f'  Number of graphs: {len(dataset)}')
            print(f'  Number of features: {dataset.num_features}')
            print(f'  Number of classes: {dataset.num_classes}')
            data = dataset[0]
            print('  --------------------')
            print(f'  Number of nodes: {data.num_nodes}')
            print(f'  Number of edges: {data.num_edges}')
            print(f'  Average node degree: {data.num_edges / data.num_nodes:.2f}')
            print(f'  Number of training nodes: {data.train_mask.sum()}')
            print(f'  Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
            print(f'  Has isolated nodes: {data.has_isolated_nodes()}')
            print(f'  Has self-loops: {data.has_self_loops()}')
            print(f'  Is undirected: {data.is_undirected()}')
            print()

    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """
        split = self.split
        self.dataset = Planetoid(root=self.data_dir, name='Cora', split=split, transform=NormalizeFeatures())
        self.loader = geom_loader.DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.workers)

    def train_dataloader(self):
        return self.loader

    def val_dataloader(self):
        return self.loader

    def test_dataloader(self):
        return self.loader

    def predict_dataloader(self):
        return self.loader
