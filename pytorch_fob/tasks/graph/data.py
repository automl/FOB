# https://github.com/Diego999/pyGAT/blob/master/utils.py

from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader as GeomDataLoader
from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.engine.utils import log_debug
from pytorch_fob.tasks import TaskDataModule


class OGBGDataModule(TaskDataModule):
    """ogbg-molhiv https://ogb.stanford.edu/docs/graphprop/#ogbg-mol
    graph: molecule, nodes: atoms, edges: chemical bonds
    features can be found here https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py
    """
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        # ogbg-molhiv is small (41,127 graphs)
        self.dataset_name = "ogbg-molhiv"
        # ogbg-molpcba is medium size (437,929 graphs)
        # self.dataset_name = "ogbg-molpcba"

    def prepare_data(self):
        dataset = PygGraphPropPredDataset(root=str(self.data_dir), name=self.dataset_name)
        log_debug(f"{dataset.num_node_features=}")
        log_debug(f"{dataset.num_classes=}")
        log_debug(f"{dataset.num_features=}")

    def get_dataloader(self, dataset, shuffle: bool = False):
        return GeomDataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
            shuffle=shuffle
        )

    def train_dataloader(self):
        self.check_dataset(self.data_train)
        return self.get_dataloader(self.data_train, shuffle=True)

    def val_dataloader(self):
        self.check_dataset(self.data_val)
        return self.get_dataloader(self.data_val)

    def test_dataloader(self):
        self.check_dataset(self.data_test)
        return self.get_dataloader(self.data_test)

    def predict_dataloader(self):
        self.check_dataset(self.data_predict)
        return self.get_dataloader(self.data_predict)

    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """
        dataset = PygGraphPropPredDataset(root=str(self.data_dir), name=self.dataset_name)
        split_idx = dataset.get_idx_split()
        if stage == "fit":
            self.data_train = dataset[split_idx["train"]]
            self.data_val = dataset[split_idx["valid"]]
        if stage == "validate":
            self.data_val = dataset[split_idx["valid"]]
        if stage == "test":
            self.data_test = dataset[split_idx["test"]]
        if stage == "predict":
            raise NotImplementedError()
