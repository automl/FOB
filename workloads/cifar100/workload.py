from typing import Any, Callable
from pathlib import Path
from lightning import LightningDataModule, LightningModule

import workloads.cifar100.data as data
import workloads.cifar100.model as model


def get_datamodule(data_dir: Path) -> LightningDataModule:
    return data.CIFAR100DataModule(data_dir)


def get_model(create_optimizer_fn: Callable) -> LightningModule:
    return model.CIFAR100Model(create_optimizer_fn)


# TODO: auxilliary information like batch_size, max_steps, etc...
def get_specs() -> dict[str, Any]:
    return dict(
        max_epochs=100
    )
