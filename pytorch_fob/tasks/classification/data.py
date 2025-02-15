from typing import Any
import numpy as np
import tensorflow_datasets as tfds
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from transformers.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from pytorch_fob.tasks import TaskDataModule
from pytorch_fob.engine.configs import TaskConfig


class Imagenet64Dataset(Dataset):
    def __init__(self, data_source) -> None:
        super().__init__()
        self.data = data_source
        self.transforms: Any

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        img = np.array(self.data[index]["image"])  # need to make copy because original is not writable
        return {"image": self.transforms(img), "label": self.data[index]["label"]}

    def set_transform(self, transforms):
        self.transforms = transforms


class ImagenetDataModule(TaskDataModule):
    def __init__(self, config: TaskConfig):
        super().__init__(config)

        self.train_transforms = self._get_train_transforms(config)

        self.val_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            v2.ToPureTensor()
        ])

    def _get_train_transforms(self, config: TaskConfig):
        # reading setting
        if config.train_transforms.random_crop.use:
            random_crop = v2.RandomCrop(
                size=config.train_transforms.random_crop.size,
                padding=config.train_transforms.random_crop.padding,
                padding_mode=config.train_transforms.random_crop.padding_mode
            )
        else:
            random_crop = v2.Identity()
        if config.train_transforms.horizontal_flip.use:
            horizontal_flip = v2.RandomHorizontalFlip(config.train_transforms.horizontal_flip.p)
        else:
            horizontal_flip = v2.Identity()
        if config.train_transforms.trivial_augment.use:
            trivial_augment = v2.TrivialAugmentWide(interpolation=v2.InterpolationMode.BILINEAR)
        else:
            trivial_augment = v2.Identity()

        transforms = v2.Compose([
            v2.ToImage(),
            random_crop,
            horizontal_flip,
            trivial_augment,
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            v2.ToPureTensor()
        ])

        return transforms

    def prepare_data(self):
        # download
        # TODO: find a solution to remove tensorflow from requirements as it is only needed for the download
        kwargs = {"file_format": "array_record"}
        tfds.data_source("imagenet_resized/64x64", data_dir=self.data_dir, download=True, builder_kwargs=kwargs)

    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """
        if stage == "fit":
            self.data_train = self._load_dataset("train")
            self.data_val = self._load_dataset("validation")
            self.data_train.set_transform(self.train_transforms)
            self.data_val.set_transform(self.val_transforms)
        if stage == "validate":
            self.data_val = self._load_dataset("validation")
            self.data_val.set_transform(self.val_transforms)
        if stage == "test":
            self.data_test = self._load_dataset("validation")
            self.data_test.set_transform(self.val_transforms)
        if stage == "predict":
            self.data_predict = self._load_dataset("validation")
            self.data_predict.set_transform(self.val_transforms)

    def _load_dataset(self, split: str) -> Imagenet64Dataset:
        ds = tfds.data_source(
            "imagenet_resized/64x64",
            split=split,
            data_dir=self.data_dir,
            download=False
        )
        rds = Imagenet64Dataset(ds)
        return rds
