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
        self.batch_size = 8
        self.train_transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.ConvertBoundingBoxFormat(tv_tensors.BoundingBoxFormat.XYXY),
                v2.SanitizeBoundingBoxes(),
                v2.ToPureTensor(),
            ]
        )
        self.val_transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.ToPureTensor(),
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
            self.data_train = self._coco_remove_images_without_annotations(
                self._wrapped_coco_dataset(train_path, annot_path / "instances_train2017.json", self.train_transforms)
            )
            self.data_val = self._wrapped_coco_dataset(val_path, annot_path / "instances_val2017.json", self.val_transforms)
        # use validation set for test and predict, because test labels are not available
        if stage == "test":
            self.data_test = self._wrapped_coco_dataset(val_path, annot_path / "instances_val2017.json", self.val_transforms)
        if stage == "predict":
            self.data_predict = self._wrapped_coco_dataset(val_path, annot_path / "instances_val2017.json", self.val_transforms)

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
        # TODO: batch_size=1 for val/test?
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.workers, collate_fn=self.collate_fn)


    def get_specs(self) -> dict[str, Any]:
        return {"batch_size": self.batch_size}

    def _wrapped_coco_dataset(self, imgs: Path, annot: Path, transforms: v2.Compose) -> CocoDetection:
        ds = CocoDetection(str(imgs), str(annot), transforms=transforms)
        return datasets.wrap_dataset_for_transforms_v2(ds, target_keys=["boxes", "labels", "image_id"])

    def _coco_img_link(self, mode: str) -> str:
        return f"http://images.cocodataset.org/zips/{mode}2017.zip"

    def _coco_annot_link(self) -> str:
        return "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    def _coco_remove_images_without_annotations(self, dataset: CocoDetection, cat_list=None):
        """
        Implementation taken from https://github.com/pytorch/vision/blob/main/references/detection/coco_utils.py
        """
        def _has_only_empty_bbox(anno):
            return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

        def _count_visible_keypoints(anno):
            return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

        min_keypoints_per_image = 10

        def _has_valid_annotation(anno):
            # if it's empty, there is no annotation
            if len(anno) == 0:
                return False
            # if all boxes have close to zero area, there is no annotation
            if _has_only_empty_bbox(anno):
                return False
            # keypoints task have a slight different criteria for considering
            # if an annotation is valid
            if "keypoints" not in anno[0]:
                return True
            # for keypoint detection tasks, only consider valid images those
            # containing at least min_keypoints_per_image
            if _count_visible_keypoints(anno) >= min_keypoints_per_image:
                return True
            return False

        ids = []
        for ds_idx, img_id in enumerate(dataset.ids):
            ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = dataset.coco.loadAnns(ann_ids)
            if cat_list:
                anno = [obj for obj in anno if obj["category_id"] in cat_list]
            if _has_valid_annotation(anno):
                ids.append(ds_idx)

        dataset = torch.utils.data.Subset(dataset, ids)
        return dataset
