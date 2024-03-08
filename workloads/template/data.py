import torch
import torch.utils.data as data
from lightning.pytorch.demos.boring_classes import RandomDataset
from runtime.configs import WorkloadConfig
from workloads import WorkloadDataModule

class TemplateDataModule(WorkloadDataModule):
    def __init__(self, workload_config: WorkloadConfig) -> None:
        super().__init__(workload_config)
        # parameters from the yaml like batch_size, name, output_dir_nam
        # are set in the super class,
        # you can set more workload specific values here
        self.split = workload_config.dataset_split_lengths

    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        ...

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        dataset = RandomDataset(1, 100)
        self.batch_size = 1
        # setting the seed here with:
        #   generator=torch.Generator().manual_seed(42)
        # is not required, as this is done with lightning.seed_everything
        self.data_train, self.data_val, self.data_test = data.random_split(
            dataset, self.split
        )
