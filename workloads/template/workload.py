from typing import Callable
from pathlib import Path
from lightning import LightningDataModule, LightningModule

import workloads.template.data as data
import workloads.template.model as model


def get_datamodule(data_dir: Path) -> LightningDataModule:
    return data.TemplateDataModule()


def get_model(create_optimizer_fn: Callable) -> LightningModule:
    return model.TemplateModel(create_optimizer_fn)


# TODO: auxilliary information like batch_size, max_steps, etc...
def get_specs(workload):
    raise NotImplementedError
