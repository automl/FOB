from pathlib import Path
from typing import Callable
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from pytorch_fob.tasks import TaskDataModule
from pytorch_fob.engine.utils import log_debug


class TabularDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        return torch.from_numpy(self.features[index]), torch.as_tensor(self.targets[index])


class TabularDataModule(TaskDataModule):
    """
    DataModule for california housing tabular data task.
    """

    def prepare_data(self):
        self.data_dir.mkdir(exist_ok=True)
        fetch_california_housing(data_home=str(self.data_dir), download_if_missing=True)
        log_debug("succesfully downloaded tabular dataset.")

    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """
        features, targets = fetch_california_housing(data_home=str(self.data_dir), return_X_y=True)
        targets = targets.astype(np.float32)  # type:ignore
        features = features.astype(np.float32)  # type:ignore

        all_idx = np.arange(len(targets))
        test_idx = self._load_test_idx()
        trainval_idx = np.setdiff1d(all_idx, test_idx)
        train_idx, val_idx = train_test_split(
            trainval_idx, train_size=self.config.train_size
        )
        assert len(train_idx) == self.config.train_size
        assert len(val_idx) == self.config.val_size
        assert len(test_idx) == self.config.test_size
        feature_preprocessor = self._get_feature_preprocessor(features[train_idx], train_idx)
        target_preprocessor = self._get_target_preprocessor(targets[train_idx])
        if stage == "fit":
            self.data_train = TabularDataset(
                feature_preprocessor(features[train_idx]),
                target_preprocessor(targets[train_idx])
            )
            self.data_val = TabularDataset(
                feature_preprocessor(features[val_idx]),
                target_preprocessor(targets[val_idx])
            )
        if stage == "validate":
            self.data_val = TabularDataset(
                feature_preprocessor(features[val_idx]),
                target_preprocessor(targets[val_idx])
            )
        if stage == "test":
            self.data_test = TabularDataset(
                feature_preprocessor(features[test_idx]),
                target_preprocessor(targets[test_idx])
            )
        if stage == "predict":
            self.data_predict = TabularDataset(
                feature_preprocessor(features[test_idx]),
                target_preprocessor(targets[test_idx])
            )

    def _load_test_idx(self) -> np.ndarray:
        # using test idx from https://github.com/yandex-research/rtdl-revisiting-models
        path = Path(__file__).resolve().parent / "idx_test.npy"
        return np.load(path)

    def _get_feature_preprocessor(self, train_features: np.ndarray, train_index: list) -> Callable:
        noise = self.config.train_transforms.noise
        if noise > 0:
            stds = np.std(train_features, axis=0, keepdims=True)
            noise_std = noise / np.maximum(stds, noise)
            noise = np.random.normal(0.0, noise_std, train_features.shape).astype(train_features.dtype)
        else:
            noise = 0.0
        if self.config.train_transforms.normalizer == "quantile":
            normalizer = QuantileTransformer(
                n_quantiles=max(min(len(train_index) // 30, 1000), 10),
                output_distribution="normal",
                subsample=10**9,
            ).fit(train_features + noise)
        elif self.config.train_transforms.normalizer == "standard":
            normalizer = StandardScaler().fit(train_features + noise)
        else:
            raise ValueError(f"Unknown normalizer {self.config.train_transforms.normalizer}")
        def preprocessor(features: np.ndarray) -> np.ndarray:
            return normalizer.transform(features)  # type:ignore
        return preprocessor

    def _get_target_preprocessor(self, train_targets: np.ndarray) -> Callable:
        tgt_mean = train_targets.mean().item()
        tgt_std  = train_targets.std().item()
        def preprocessor(targets: np.ndarray) -> np.ndarray:
            return (targets - tgt_mean) / tgt_std
        return preprocessor
