from engine.configs import TaskConfig
from optimizers import Optimizer
from tasks import TaskModel, TaskDataModule
from tasks.ogbg import data
from tasks.ogbg import model


def get_datamodule(config: TaskConfig) -> TaskDataModule:
    return data.OGBGDataModule(config)

def get_task(optimizer: Optimizer, config: TaskConfig) -> tuple[TaskModel, TaskDataModule]:
    datamodule = data.OGBGDataModule(config)
    node_feature_dim = datamodule.feature_dim
    num_classes = datamodule.num_classes
    dataset_name = datamodule.dataset_name
    ogbg_model = model.OGBGModel(optimizer,
                                 node_feature_dim=node_feature_dim,
                                 num_classes=num_classes,
                                 dataset_name=dataset_name,
                                 batch_size=datamodule.batch_size,
                                 config=config)
    return ogbg_model, datamodule
