from typing import Any
from urllib import request
from pathlib import Path
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms
from workloads import WorkloadDataModule


class COCODataModule(WorkloadDataModule):
    def __init__(self, data_dir: Path):
        super().__init__()
        self.data_dir = data_dir / "COCO"
        self.batch_size = 512

    def prepare_data(self):
        self.data_dir.mkdir(exist_ok=True)
        # download images
        dl_dir = self.data_dir / "downloads"
        dl_dir.mkdir(exist_ok=True)
        link = self.coco_annot_link()
        request.urlretrieve(link, dl_dir / link.rsplit('/', maxsplit=1)[-1])

    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass

    def get_specs(self) -> dict[str, Any]:
        return {"batch_size": self.batch_size}

    def coco_img_link(self, mode: str) -> str:
        return f"http://images.cocodataset.org/zips/{mode}2017.zip"

    def coco_annot_link(self) -> str:
        return "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
