from pathlib import Path
from typing import Optional

import numpy as np
import torch
from google.auth.exceptions import GoogleAuthError
from google.cloud import storage
from torch.utils.data import Dataset, Subset
from torchvision.transforms import v2
from tqdm import tqdm

from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.engine.utils import log_debug, log_info, log_warn
from pytorch_fob.tasks import TaskDataModule


class TransformSubset(Subset):
    def __init__(self, dataset, indices, transform=None, target_transform=None):
        super().__init__(dataset, indices)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y


class CIFAR5MDataset(Dataset):
    def __init__(self, npz_files: list[str], preload: bool = False):
        """
        Args:
            npz_files: List of paths to .npz files
            preload: If True, load all data into memory
        """
        self.npz_files = npz_files
        self.preload = preload

        # Calculate lengths and offsets for each file
        self.file_lengths = []
        self.file_offsets = [0]

        for f in npz_files:
            with np.load(f) as data:
                length = len(data["Y"])
                self.file_lengths.append(length)
                self.file_offsets.append(self.file_offsets[-1] + length)


        self.total_length = self.file_offsets[-1]
        log_debug(f"File offsets: {self.file_offsets}")
        log_debug(f"Total length: {self.total_length}")

        # Optionally preload all data
        self._preloaded_data = None
        if preload:
            log_info("Preloading data...")
            x_data = np.empty((self.total_length, 32, 32, 3), dtype=np.uint8)
            y_data = np.empty(self.total_length, dtype=np.int64)

            offset = 0
            for f in tqdm(npz_files, desc="Reading data from files", unit="file"):
                with np.load(f) as data:
                    sample_count = len(data["Y"])
                    x_data[offset : offset + sample_count] = data["X"]
                    y_data[offset : offset + sample_count] = data["Y"]
            self._preloaded_data = {"X": x_data, "Y": y_data}
            log_info("Preloading complete.")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if self._preloaded_data is not None:
            # Get from preloaded data
            x, y = self._preloaded_data["X"][idx], self._preloaded_data["Y"][idx]
        else:
            # Find which file the index belongs to
            file_idx = 0
            while file_idx < len(self.file_offsets) - 1 and idx >= self.file_offsets[file_idx + 1]:
                file_idx += 1

            # Calculate the local index within the file
            local_idx = idx - self.file_offsets[file_idx]

            # Load from the appropriate file
            with np.load(self.npz_files[file_idx]) as data:
                x, y = data["X"][local_idx], data["Y"][local_idx]

        # Convert to torch tensors
        x = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0  # CHW format and normalize
        y = torch.tensor(y, dtype=torch.long)

        return x, y


class CIFAR5MDataModule(TaskDataModule):
    def __init__(self, config: TaskConfig):
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
        self.val_transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float, scale=True),
                v2.Normalize(cifar100_mean, cifar100_stddev),
                v2.ToPureTensor(),
            ]
        )

        self.nsamples = 6002688
        self.max_train_size = 5002240
        self.max_val_size = 1000448
        assert self.config.train_size <= self.max_train_size, (
            "Train size must be less than total number of samples."
        )
        assert self.config.val_size <= self.max_val_size, (
            "Val size must be less than total number of samples."
        )
        self._full_ds = None
        self._indices = None

    def prepare_data(self):
        # download
        dl_path = self.data_dir / "download"
        dl_path.mkdir(parents=True, exist_ok=True)
        try:
            if len(list(dl_path.iterdir())) != 6:
                self._download(dl_path)
        except GoogleAuthError as e:
            log_warn("Google auth error. Skipping download. Please authenticate and try again.")
            log_warn(e.args[0])

    def _download(self, path: Path):
        client = storage.Client()
        bucket = client.get_bucket("gresearch")
        blobs = bucket.list_blobs(prefix="cifar5m/")
        for blob in blobs:
            print(f"Downloading {blob.name}")
            blob.download_to_filename(path / blob.name)

    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended."""
        self._init_dataset(preload=True)
        if stage == "fit":
            self.data_train = self._get_dataset(train=True)
            self.data_val = self._get_dataset(train=False)

        if stage == "validate":
            self.data_val = self._get_dataset(train=False)

        if stage == "test":
            self.data_test = self._get_dataset(train=False, size=self.max_val_size)

        if stage == "predict":
            self.data_predict = self._get_dataset(train=False, size=self.max_val_size)

    def _get_dataset(self, train: bool, size: Optional[int] = None) -> Subset:
        if self._indices is None:
            self._indices = torch.randperm(self.nsamples, generator=torch.Generator().manual_seed(42)).tolist()

        if train:
            size = size if size is not None else self.config.train_size
            ds = TransformSubset(self._full_ds, self._indices[:size], transform=self.train_transforms)
        else:
            size = -size if size is not None else -self.config.val_size
            ds = TransformSubset(self._full_ds, self._indices[size:], transform=self.val_transforms)
        return ds

    def _init_dataset(self, preload: bool = False):
        if self._full_ds is None or (preload and not self._full_ds.preload):
            npz_files = list(map(str, (self.data_dir / "download").iterdir()))
            self._full_ds = CIFAR5MDataset(
                npz_files,
                preload=preload,
            )
