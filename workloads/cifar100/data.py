from typing import Any
from pathlib import Path
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms
from workloads import WorkloadDataModule
from bob.runtime import RuntimeArgs


class CIFAR100DataModule(WorkloadDataModule):
    def __init__(self, runtime_args: RuntimeArgs):
        super().__init__(runtime_args)
        self.batch_size = 1024  # TODO: which batch size to use???
        # cifar 100 has 60000 32x32 color images (600 images per class)
        # 10k for test
        self.train_val_split = [45000, 5000]
        self.seed = 42

        # TODO check values
        # https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151?permalink_comment_id=2851662#gistcomment-2851662
        meanOfCIFAR100 = torch.tensor([0.5071, 0.4865, 0.4409])
        stdOfCIFAR100 = torch.tensor([0.2673, 0.2564, 0.2762])
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(meanOfCIFAR100, stdOfCIFAR100)])

    def prepare_data(self):
        # download
        CIFAR100(str(self.data_dir), train=True, download=True)
        CIFAR100(str(self.data_dir), train=False, download=True)
    
    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            cifar100_full = CIFAR100(str(self.data_dir), train=True, transform=self.transform)
            gen = torch.Generator().manual_seed(self.seed)
            self.data_train, self.data_val = random_split(cifar100_full, self.train_val_split, generator=gen)
        
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.data_test = CIFAR100(str(self.data_dir), train=False, transform=self.transform)

        if stage == "predict":
            self.data_predict = CIFAR100(str(self.data_dir), train=False, transform=self.transform)

    def get_specs(self) -> dict[str, Any]:
        return {"batch_size": self.batch_size}
    