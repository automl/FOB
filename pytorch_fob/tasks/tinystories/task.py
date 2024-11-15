from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.optimizers import Optimizer
from pytorch_fob.tasks import TaskDataModule, TaskModel
from pytorch_fob.tasks.tinystories.data import TinyStoriesDataModule
from pytorch_fob.tasks.tinystories.model import GPTModel


def get_datamodule(config: TaskConfig) -> TaskDataModule:
    return TinyStoriesDataModule(config)


def get_task(optimizer: Optimizer, config: TaskConfig) -> tuple[TaskModel, TaskDataModule]:
    dm = get_datamodule(config)
    vs = dm.get_vocab_size()  # type: ignore
    return GPTModel(optimizer, config, vocab_size=vs), dm
