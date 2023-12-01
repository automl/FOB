from lightning import LightningDataModule
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms


class CIFAR10DataModule(LightningDataModule):
    def __init__(self, data_dir: str = "./data"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = 32  # TODO: which batch size to use???
        # cifar 10 has 60000 32x32 color images (600 images per class)
        # 10k for testing
        self.train_val_split = [45000, 5000]
        self.seed = 42

        # TODO check values
        # https://github.com/facebookarchive/fb.resnet.torch/issues/180
        meanOfCIFAR10 = torch.tensor([0.4913997551666284, 0.48215855929893703, 0.4465309133731618])
        stdOfCIFAR10 = torch.tensor([0.24703225141799082, 0.24348516474564, 0.26158783926049628])
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(meanOfCIFAR10, stdOfCIFAR10)])

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)
    
    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            cifar10_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            gen = torch.Generator().manual_seed(self.seed)
            self.cifar10_train, self.cifar10_val = random_split(cifar10_full, self.train_val_split, generator=gen)
        
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.cifar10_predict = CIFAR10(self.data_dir, train=False, transform=self.transform)


    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.cifar10_predict, batch_size=self.batch_size)
    