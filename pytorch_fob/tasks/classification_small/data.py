import torch
from torchvision.datasets import CIFAR100
from torchvision.transforms import v2

from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.tasks import TaskDataModule


class CIFAR100DataModule(TaskDataModule):
    def __init__(self, config: TaskConfig):
        # cifar 100 has 60000 32x32 color images (600 images per class)
        super().__init__(config)
        cifar100_mean = (0.4914, 0.4822, 0.4465)
        cifar100_stddev = (0.2023, 0.1994, 0.2010)
        # build the transforms as given in config
        random_crop = (
            v2.RandomCrop(
                size=config.train_transforms.random_crop.size,
                padding=config.train_transforms.random_crop.padding,
                padding_mode=config.train_transforms.random_crop.padding_mode,
            )
            if config.train_transforms.random_crop.use
            else v2.Identity()
        )
        horizontal_flip = (
            v2.RandomHorizontalFlip(config.train_transforms.horizontal_flip.p)
            if config.train_transforms.horizontal_flip.use
            else v2.Identity()
        )
        trivial_augment = (
            v2.TrivialAugmentWide(interpolation=v2.InterpolationMode.BILINEAR)
            if config.train_transforms.trivial_augment.use
            else v2.Identity()
        )
        label_flipping = (
            LabelFlipping(config.train_transforms.label_flipping.p, 100)
            if config.train_transforms.label_flipping.use
            else v2.Identity()
        )

        self.train_transforms = v2.Compose(
            [
                v2.ToImage(),
                random_crop,
                horizontal_flip,
                trivial_augment,
                v2.ToDtype(torch.float, scale=True),
                v2.Normalize(cifar100_mean, cifar100_stddev),
                v2.ToPureTensor(),
            ]
        )
        self.train_target_transforms = label_flipping
        self.val_transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float, scale=True),
                v2.Normalize(cifar100_mean, cifar100_stddev),
                v2.ToPureTensor(),
            ]
        )

    def prepare_data(self):
        # download
        CIFAR100(str(self.data_dir), train=True, download=True)
        CIFAR100(str(self.data_dir), train=False, download=True)

    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended."""
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
            return CIFAR100(
                str(self.data_dir),
                train=True,
                transform=self.train_transforms,
                target_transform=self.train_target_transforms,
            )
        else:
            return CIFAR100(str(self.data_dir), train=False, transform=self.val_transforms)


class LabelFlipping:
    """
    Flips the label to a random label with a given probability.
    """

    def __init__(self, flip_probability: float = 0.1, num_classes: int = 100):
        self.flip_probability = flip_probability
        self.num_classes = num_classes

    def __call__(self, target):
        if torch.rand(1) < self.flip_probability:
            # Generate new random label that's different from target
            new_label = torch.randint(0, self.num_classes, (1,))[0]
            while new_label == target:
                new_label = torch.randint(0, self.num_classes, (1,))[0]
            return new_label.item()
        return target
