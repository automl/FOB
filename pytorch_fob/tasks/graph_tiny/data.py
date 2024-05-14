import torch_geometric.loader as geom_loader
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from pytorch_fob.tasks import TaskDataModule
from pytorch_fob.engine.configs import TaskConfig


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
        self.data_dir.mkdir(exist_ok=True, parents=True)

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

        Planetoid(root=str(self.data_dir), name='Cora', split=self.split, transform=NormalizeFeatures())

    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        for this task the forward pass will use masks and
        only calculate the loss on the nodes corresponding to the mask
        """
        dataset = Planetoid(root=str(self.data_dir), name='Cora', split=self.split, transform=NormalizeFeatures())
        if stage == "fit":
            self.data_train = dataset
            self.data_val = dataset
        elif stage == "validate":
            self.data_val = dataset
        elif stage == "test":
            self.data_test = dataset
        elif stage == "predict":
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def get_dataloader(self, dataset):
        return geom_loader.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.workers)

    def train_dataloader(self):
        return self.get_dataloader(self.data_train)

    def val_dataloader(self):
        return self.get_dataloader(self.data_val)

    def test_dataloader(self):
        return self.get_dataloader(self.data_test)

    def predict_dataloader(self):
        return self.get_dataloader(self.data_predict)
