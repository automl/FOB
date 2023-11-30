import lightning as L
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = 32
        # split can also be a fraction self.train_val_split
        # [55000, 5000] is taken from https://lightning.ai/docs/pytorch/stable/data/datamodule.html
        self.train_val_split = [55000, 5000]
        self.seed = 42

        # TODO: which transform should we use here https://lightning.ai/docs/pytorch/stable/data/datamodule.html
        # (0.1307,), (0.3081,) is taken from https://lightning.ai/docs/pytorch/stable/data/datamodule.html
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)
    
    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, self.train_val_split, generator=torch.Generator().manual_seed(self.seed)
            )
        
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)


    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size==self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size==self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size==self.batch_size)