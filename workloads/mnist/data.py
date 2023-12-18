from typing import Any
from pathlib import Path
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from workloads import WorkloadDataModule


class MNISTDataModule(WorkloadDataModule):
    def __init__(self, data_dir: Path):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = 512
        # split can also be a fraction self.train_val_split
        # [55000, 5000] is taken from https://lightning.ai/docs/pytorch/stable/data/datamodule.html
        self.train_val_split = [55000, 5000]
        self.seed = 42

        # TODO: check values
        # https://lightning.ai/docs/pytorch/stable/data/datamodule.html
        meanOfMNIST = (0.1307,)
        stdOfMNIST = (0.3081,)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(meanOfMNIST, stdOfMNIST)])
    
    def prepare_data(self):
        # download
        MNIST(str(self.data_dir), train=True, download=True)
        MNIST(str(self.data_dir), train=False, download=True)
    
    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(str(self.data_dir), train=True, transform=self.transform)
            gen = torch.Generator().manual_seed(self.seed)
            self.data_train, self.data_val = random_split(mnist_full, self.train_val_split, generator=gen)
        
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.data_test = MNIST(str(self.data_dir), train=False, transform=self.transform)

        if stage == "predict":
            self.data_predict = MNIST(str(self.data_dir), train=False, transform=self.transform)

    def get_specs(self) -> dict[str, Any]:
        return {"batch_size": self.batch_size}
    