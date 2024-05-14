from torch.utils import data
from lightning.pytorch.demos.boring_classes import RandomDataset
from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.tasks import TaskDataModule


class TemplateDataModule(TaskDataModule):
    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        # parameters from the yaml like batch_size, name, output_dir_nam
        # are set in the super class,
        # you can set more task specific values here
        self.split = config.dataset_split_lengths

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
