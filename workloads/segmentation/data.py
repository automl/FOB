import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoImageProcessor
from torchvision.transforms import v2
from workloads import WorkloadDataModule
from runtime import DatasetArgs


class SegmentationDataModule(WorkloadDataModule):
    """
    DataModule for SceneParse150 semantic segmentation task.
    """
    def __init__(self, dataset_args: DatasetArgs):
        super().__init__(dataset_args)
        self.data_dir = self.data_dir / "SceneParse150"
        self.batch_size = 4
        tv_train_transforms = v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
        tv_val_transforms = v2.Identity()
        image_processor = AutoImageProcessor.from_pretrained("nvidia/mit-b0", do_reduce_labels=True)
        def train_transforms(batch):
            images = list(map(tv_train_transforms, batch["image"]))
            targets = batch["annotation"]
            return image_processor(images, targets)
        def val_transforms(batch):
            images = list(map(tv_val_transforms, batch["image"]))
            targets = batch["annotation"]
            return image_processor(images, targets)

        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

    def prepare_data(self):
        self.data_dir.mkdir(exist_ok=True)
        load_dataset("scene_parse_150", cache_dir=str(self.data_dir), trust_remote_code=True)

    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """
        if stage == "fit":
            self.data_train = self._load_dataset("train")
            self.data_val = self._load_dataset("validation")
            self.data_train.set_transform(self.train_transforms)
            self.data_val.set_transform(self.val_transforms)
        if stage == "validate":
            self.data_val = self._load_dataset("validation")
            self.data_val.set_transform(self.val_transforms)
        if stage == "test":
            # no labels available for test split, so we use val
            self.data_test = self._load_dataset("validation")
            self.data_test.set_transform(self.val_transforms)
        if stage == "predict":
            self.data_predict = self._load_dataset("test")
            self.data_predict.set_transform(self.val_transforms)

    def _load_dataset(self, split: str) -> Dataset:
        ds = load_dataset(
            "scene_parse_150",
            cache_dir=str(self.data_dir),
            split=split,
            trust_remote_code=True
        )
        return self._remove_invalid_images(ds) # type:ignore

    def _remove_invalid_images(self, dataset: Dataset) -> Dataset:
        return dataset.filter(lambda d: np.array(d["image"]).ndim >= 3)
