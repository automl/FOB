from typing import Callable
from pathlib import Path
from lightning import LightningDataModule, LightningModule

import workloads.cifar100.data as data
import workloads.cifar100.model as model


def get_datamodule(data_dir: Path) -> LightningDataModule:
    return data.CIFAR100DataModule(data_dir)


def get_model(create_optimizer_fn: Callable) -> LightningModule:
    return model.CIFAR100Model(create_optimizer_fn)


# TODO: auxilliary information like batch_size, max_steps, etc...
def get_specs(workload):
    raise NotImplementedError
    # make sure to use the same batch_size that is also used in the data module
    data.CIFAR100DataModuleDataModule("path").batch_size
