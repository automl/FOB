from pathlib import Path
import zipfile
import wget
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2
from torchvision import datasets, tv_tensors
from pycocotools.coco import COCO
from tqdm import tqdm
from pytorch_fob.tasks import TaskDataModule
from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.engine.utils import log_info


class COCODataModule(TaskDataModule):
    """
    DataModule for COCO object detection task.
    Implementation and choice of transforms is heavily inspired by
    https://github.com/pytorch/vision/tree/main/references/detection
    """
    def __init__(self, config: TaskConfig):
        super().__init__(config)

        if config.train_transforms.horizontal_flip.use:
            horizontal_flip = v2.RandomHorizontalFlip(p=config.train_transforms.horizontal_flip.p)
        else:
            horizontal_flip = v2.Identity()

        self.train_transforms = v2.Compose(
            [
                v2.ToImage(),
                horizontal_flip,
                v2.ToDtype(torch.float, scale=True),
                v2.ConvertBoundingBoxFormat(tv_tensors.BoundingBoxFormat.XYXY),
                v2.SanitizeBoundingBoxes(),
                v2.ToPureTensor(),
            ]
        )
        self.val_transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float, scale=True),
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
            log_info(f"{subject} already downloaded.")
        else:
            log_info(f"Downloading {subject}...")
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
                log_info(f"{subject} already extracted.")
                return
        with zipfile.ZipFile(file, "r") as f:
            for img in tqdm(f.infolist(), desc=f"Extracting {subject}"):
                f.extract(img, out)
        with open(state_file, "a", encoding="utf8") as f:
            f.write(f"{subject}\n")

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
            self.data_val = self._coco_remove_images_without_annotations(
                self._wrapped_coco_dataset(val_path, annot_path / "instances_val2017.json", self.val_transforms)
            )
        # use validation set for test and predict, because test labels are not available
        if stage == "validate":
            self.data_val = self._wrapped_coco_dataset(
                val_path, annot_path / "instances_val2017.json", self.val_transforms
            )
        if stage == "test":
            self.data_test = self._wrapped_coco_dataset(
                val_path, annot_path / "instances_val2017.json", self.val_transforms
            )
        if stage == "predict":
            self.data_predict = self._wrapped_coco_dataset(
                val_path, annot_path / "instances_val2017.json", self.val_transforms
            )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader_from_dataset(self.data_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader_from_dataset(self.data_val, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader_from_dataset(self.data_test, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self._dataloader_from_dataset(self.data_predict)

    def _dataloader_from_dataset(
            self,
            dataset: Dataset,
            batch_size: int | None = None,
            shuffle: bool = False
        ) -> DataLoader:
        if batch_size is None:
            batch_size = self.batch_size
        self.check_dataset(dataset)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
            shuffle=shuffle
        )

    def eval_gt_data(self) -> COCO:
        val_path = self.data_dir / "val2017"
        annot_path = self.data_dir / "annotations" / "instances_val2017.json"
        ds = self._wrapped_coco_dataset(val_path, annot_path, transforms=self.val_transforms)
        return ds.coco

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
