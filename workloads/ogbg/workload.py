from runtime.configs import WorkloadConfig
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.ogbg import data
from workloads.ogbg import model


def get_datamodule(workload_config: WorkloadConfig) -> WorkloadDataModule:
    return data.OGBGDataModule(workload_config)

def get_workload(submission: Submission, workload_config: WorkloadConfig) -> tuple[WorkloadModel, WorkloadDataModule]:
    datamodule = data.OGBGDataModule(workload_config)
    node_feature_dim = datamodule.feature_dim
    num_classes = datamodule.num_classes
    dataset_name = datamodule.dataset_name
    ogbg_model = model.OGBGModel(submission,
                                 node_feature_dim=node_feature_dim,
                                 num_classes=num_classes,
                                 dataset_name=dataset_name,
                                 batch_size=datamodule.batch_size,
                                 workload_config=workload_config)
    return ogbg_model, datamodule
