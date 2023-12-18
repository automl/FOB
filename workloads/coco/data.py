from typing import Any
from pathlib import Path
import zipfile
import wget
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2
from torchvision import datasets, tv_tensors
from tqdm import tqdm
from workloads import WorkloadDataModule
from bob.runtime import RuntimeArgs


class COCODataModule(WorkloadDataModule):
    """
    DataModule for COCO object detection task.
    Implementation and choice of transforms is heavily inspired by example in
    https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_e2e.html
    """
    def __init__(self, runtime_args: RuntimeArgs):
        super().__init__(runtime_args)
        self.data_dir = self.data_dir / "COCO"
        self.batch_size = 2
        self.transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.RandomPhotometricDistort(p=1),
                v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
                v2.RandomIoUCrop(),
                v2.RandomHorizontalFlip(p=1),
                v2.SanitizeBoundingBoxes(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        self.collate_fn = lambda batch: tuple(zip(*batch))

    def prepare_data(self):
        self.data_dir.mkdir(exist_ok=True)
        # download images
        annot = self._download(self._coco_annot_link(), "annotations")
        train = self._download(self._coco_img_link("train"), "train images")
        val = self._download(self._coco_img_link("val"), "val images")
        # no annotations for test set available, so no need to download
        # test = self._download(self._coco_img_link("test"), "test images")
        # extract
        self._extract(annot, self.data_dir, "annotations")
        self._extract(train, self.data_dir, "train images")
        self._extract(val, self.data_dir, "val images")
        # self._extract(test, self.data_dir, "test images")
        # TODO: cleanup zip files, we should have an argument for this


    def _download(self, url: str, subject: str) -> Path:
        dl_dir = self.data_dir / "downloads"
        dl_dir.mkdir(exist_ok=True)
        filename = wget.detect_filename(url)
        outfile = dl_dir / filename
        if outfile.exists():
            print(f"{subject} already downloaded.")
        else:
            print(f"Downloading {subject}...")
            try:
                wget.download(url, str(outfile))
            except BaseException as e:
                self._clean_tmpfiles(dl_dir)
                raise e
        return outfile

    def _clean_tmpfiles(self, path: Path):
        for f in path.glob("*.tmp"):
            f.unlink()

    def _extract(self, file: Path, out: Path, subject: str):
        state_file = out / "extracted_subjects.txt"
        if not state_file.exists():
            state_file.touch()
        with open(state_file, "r", encoding="utf8") as f:
            extracted_subjects = f.readlines()
            if subject in map(lambda s: s.strip(), extracted_subjects):
                print(f"{subject} already extracted.")
                return
        with zipfile.ZipFile(file, "r") as f:
            for img in tqdm(f.infolist(), desc=f"Extracting {subject}"):
                f.extract(img, out)
        with open(state_file, "a", encoding="utf8") as f:
            f.write(subject)

    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """
        train_path = self.data_dir / "train2017"
        val_path = self.data_dir / "val2017"
        annot_path = self.data_dir / "annotations"
        if stage == "fit":
            self.data_train = self._wrapped_coco_dataset(train_path, annot_path / "instances_train2017.json")
            self.data_val = self._wrapped_coco_dataset(val_path, annot_path / "instances_val2017.json")
        # use validation set for test and predict, because test labels are not available
        if stage == "test":
            self.data_test = self._wrapped_coco_dataset(val_path, annot_path / "instances_val2017.json")
        if stage == "predict":
            self.data_predict = self._wrapped_coco_dataset(val_path, annot_path / "instances_val2017.json")

    def train_dataloader(self) -> DataLoader:
        return self._dataloader_from_dataset(self.data_train)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader_from_dataset(self.data_val)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader_from_dataset(self.data_test)

    def predict_dataloader(self) -> DataLoader:
        return self._dataloader_from_dataset(self.data_predict)

    def _dataloader_from_dataset(self, dataset: Dataset) -> DataLoader:
        self.check_dataset(dataset)
        return DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)


    def get_specs(self) -> dict[str, Any]:
        return {"batch_size": self.batch_size}

    def _wrapped_coco_dataset(self, imgs: Path, annot: Path) -> Dataset:
        ds = CocoDetection(str(imgs), str(annot), transforms=self.transforms)
        return datasets.wrap_dataset_for_transforms_v2(ds, target_keys=["boxes", "labels", "masks"])

    def _coco_img_link(self, mode: str) -> str:
        return f"http://images.cocodataset.org/zips/{mode}2017.zip"

    def _coco_annot_link(self) -> str:
        return "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
