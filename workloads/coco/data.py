from typing import Any
from pathlib import Path
from tqdm import tqdm
import wget
import zipfile
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms
from workloads import WorkloadDataModule
from bob.runtime import RuntimeArgs


class COCODataModule(WorkloadDataModule):
    def __init__(self, runtime_args: RuntimeArgs):
        super().__init__(runtime_args)
        self.data_dir = self.data_dir / "COCO"
        self.batch_size = 512

    def prepare_data(self):
        self.data_dir.mkdir(exist_ok=True)
        # download images
        annot = self._download(self._coco_annot_link(), "annotations")
        train = self._download(self._coco_img_link("train"), "train images")
        val = self._download(self._coco_img_link("val"), "val images")
        test = self._download(self._coco_img_link("test"), "test images")
        # extract
        self._extract(annot, self.data_dir, "annotations")
        self._extract(train, self.data_dir, "train images")
        self._extract(val, self.data_dir, "val images")
        self._extract(test, self.data_dir, "test images")


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
        pass

    def get_specs(self) -> dict[str, Any]:
        return {"batch_size": self.batch_size}

    def _coco_img_link(self, mode: str) -> str:
        return f"http://images.cocodataset.org/zips/{mode}2017.zip"

    def _coco_annot_link(self) -> str:
        return "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
