from typing import Any
import torch
from torchvision.datasets import CIFAR100
from torchvision.transforms import v2
from workloads import WorkloadDataModule
from bob.runtime import DatasetArgs


class CIFAR100DataModule(WorkloadDataModule):
    def __init__(self, runtime_args: DatasetArgs):
        # cifar 100 has 60000 32x32 color images (600 images per class)
        super().__init__(runtime_args)
        self.batch_size = 128
        cifar100_mean = (0.4914, 0.4822, 0.4465)
        cifar100_stddev = (0.2023, 0.1994, 0.2010)
        self.train_transforms = v2.Compose([
            v2.ToImage(),
            v2.RandomCrop(32, padding=4, padding_mode='reflect'),
            v2.RandomHorizontalFlip(),
            v2.TrivialAugmentWide(interpolation=v2.InterpolationMode.BILINEAR),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(cifar100_mean, cifar100_stddev),
            v2.ToPureTensor()
        ])
        self.val_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(cifar100_mean, cifar100_stddev),
            v2.ToPureTensor()
        ])

    def prepare_data(self):
        # download
        CIFAR100(str(self.data_dir), train=True, download=True)
        CIFAR100(str(self.data_dir), train=False, download=True)

    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """
        if stage == "fit":
            self.data_train = self._get_dataset(train=True)
            self.data_val = self._get_dataset(train=False)

        if stage == "validate":
            self.data_val = self._get_dataset(train=False)

        if stage == "test":
            self.data_test = self._get_dataset(train=False)

        if stage == "predict":
            self.data_predict = self._get_dataset(train=False)

    def _get_dataset(self, train: bool):
        if train:
            return CIFAR100(str(self.data_dir), train=True, transform=self.train_transforms)
        else:
            return CIFAR100(str(self.data_dir), train=False, transform=self.val_transforms)
