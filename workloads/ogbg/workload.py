from lightning import Callback
from runtime import DatasetArgs
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.ogbg import data
from workloads.ogbg import model


def get_datamodule(dataset_args: DatasetArgs) -> WorkloadDataModule:
    return data.OGBGDataModule(dataset_args)


def get_workload(submission: Submission, dataset_args: DatasetArgs) -> tuple[WorkloadModel, WorkloadDataModule]:
    datamodule = data.OGBGDataModule(dataset_args)
    node_feature_dim = datamodule.feature_dim
    num_classes = datamodule.num_classes
    return model.OGBGModel(submission, node_feature_dim, num_classes), datamodule

def get_callbacks() -> list[Callback]:
    return []
