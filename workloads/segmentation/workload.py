from lightning import Callback
from runtime import DatasetArgs
from submissions import Submission
from workloads import WorkloadModel, WorkloadDataModule
from workloads.segmentation.data import SegmentationDataModule
from workloads.segmentation.model import SegmentationModel


def get_datamodule(dataset_args: DatasetArgs) -> WorkloadDataModule:
    return SegmentationDataModule(dataset_args)


def get_workload(submission: Submission, dataset_args: DatasetArgs) -> tuple[WorkloadModel, WorkloadDataModule]:
    dm = SegmentationDataModule(dataset_args)
    return SegmentationModel(submission, dm.data_dir, dm.id2label, dm.label2id), dm

def get_callbacks() -> list[Callback]:
    return []
