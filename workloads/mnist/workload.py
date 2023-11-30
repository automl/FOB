from typing import Callable
from pathlib import Path
from lightning import LightningDataModule, LightningModule

import workloads.mnist.data as data
import workloads.mnist.model as model

def get_datamodule(data_dir: Path) -> LightningDataModule:
    return data.MNISTDataModule(data_dir)

def get_model(create_optimizer_fn: Callable) -> LightningModule:
    return model.MNISTModel(create_optimizer_fn)

# TODO: auxilliary information like batch_size, max_steps, etc...
def get_specs():
    raise NotImplementedError
