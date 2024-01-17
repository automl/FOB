from lightning import Callback
from runtime import DatasetArgs
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.segmentation.data import SegmentationDataModule
from workloads.segmentation.model import SegmentationModel


def get_datamodule(dataset_args: DatasetArgs) -> WorkloadDataModule:
    return SegmentationDataModule(dataset_args)


def get_workload(submission: Submission, dataset_args: DatasetArgs) -> tuple[WorkloadModel, WorkloadDataModule]:
    return SegmentationModel(submission), SegmentationDataModule(dataset_args)

def get_callbacks() -> list[Callback]:
    return []
