from typing import Callable
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from workloads import WorkloadDataModule
from engine.configs import WorkloadConfig


class TabularDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        return torch.from_numpy(self.features[index]), torch.as_tensor(self.targets[index])


class TabularDataModule(WorkloadDataModule):
    """
    DataModule for california housing tabular data task.
    """
    def __init__(self, workload_config: WorkloadConfig):
        super().__init__(workload_config)
        self.train_split = workload_config.train_split

    def prepare_data(self):
        self.data_dir.mkdir(exist_ok=True)
        fetch_california_housing(data_home=str(self.data_dir), download_if_missing=True)
        print("succesfully downloaded tabular dataset.")

    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """
        features, targets = fetch_california_housing(data_home=str(self.data_dir), return_X_y=True)
        targets = targets.astype(np.float32)  # type:ignore
        features = features.astype(np.float32)  # type:ignore

        all_idx = np.arange(len(targets))
        trainval_idx, test_idx = train_test_split(
            all_idx, train_size=self.train_split
        )
        train_idx, val_idx = train_test_split(
            trainval_idx, train_size=self.train_split
        )
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

    def _get_feature_preprocessor(self, train_features: np.ndarray, train_index: list) -> Callable:
        noise = (
            np.random.default_rng(0)
            .normal(0.0, 1e-5, train_features.shape)
            .astype(train_features.dtype)
        )
        qt = QuantileTransformer(
            n_quantiles=max(min(len(train_index) // 30, 1000), 10),
            output_distribution="normal",
            subsample=10**9,
        ).fit(train_features + noise)
        def preprocessor(features: np.ndarray) -> np.ndarray:
            return qt.transform(features)  # type:ignore
        return preprocessor

    def _get_target_preprocessor(self, train_targets: np.ndarray) -> Callable:
        tgt_mean = train_targets.mean().item()
        tgt_std  = train_targets.std().item()
        def preprocessor(targets: np.ndarray) -> np.ndarray:
            return (targets - tgt_mean) / tgt_std
        return preprocessor
