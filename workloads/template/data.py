from typing import Any
import torch
import torch.utils.data as data
from lightning.pytorch.demos.boring_classes import RandomDataset
from workloads import WorkloadDataModule

class TemplateDataModule(WorkloadDataModule):
    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        ...

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        dataset = RandomDataset(1, 100)
        self.batch_size = 1
        self.data_train, self.data_val, self.data_test = data.random_split(
            dataset, [80, 10, 10], generator=torch.Generator().manual_seed(42)
        )

    def get_specs(self) -> dict[str, Any]:
        return {"batch_size": 42}
