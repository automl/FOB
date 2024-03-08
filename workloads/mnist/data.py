from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from runtime.configs import WorkloadConfig
from workloads import WorkloadDataModule


class MNISTDataModule(WorkloadDataModule):
    def __init__(self, workload_config: WorkloadConfig):
        super().__init__(workload_config)
        # split can also be a fraction self.train_val_split
        # [55000, 5000] is taken from https://lightning.ai/docs/pytorch/stable/data/datamodule.html
        self.train_val_split = [55000, 5000]

        # TODO: check values
        # https://lightning.ai/docs/pytorch/stable/data/datamodule.html
        mean = (0.1307,)
        std = (0.3081,)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])

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
            # TODO (Zachi) confirm seed everything makes this reproducable:
            self.data_train, self.data_val = random_split(mnist_full, self.train_val_split)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.data_test = MNIST(str(self.data_dir), train=False, transform=self.transform)

        if stage == "predict":
            self.data_predict = MNIST(str(self.data_dir), train=False, transform=self.transform)
