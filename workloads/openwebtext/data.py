# https://huggingface.co/datasets/Skylion007/openwebtext

# https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/builder_classes#datasets.DownloadManager
# downloaded_files = dl_manager.download('https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz')
# extracted_files = dl_manager.extract(downloaded_files)

from typing import Any
import torch
from torch.utils.data import DataLoader
# from torchvision import transforms
from workloads import WorkloadDataModule

class OpenWebTextDataModule(WorkloadDataModule):
    def __init__(self, data_dir: str = "./data"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = 1  # TODO
        self.train_val_split = [1, 1]  # TODO
        self.seed = 42

        # TODO do we need to normalize?
        # meanOfOpenWebText = torch.tensor(0)
        # stdOfOpenWebText = torch.tensor(1)
        # self.transform  = transforms.Compose([transforms.ToTensor(),transforms.Normalize(meanOfOpenWebText, stdOfOpenWebText)])  
        
    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if stage == "fit":
            pass
        if stage == "test":
            pass
        if stage == "predict":
            pass

    def train_dataloader(self):
        return DataLoader(self.openwebtext_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.openwebtext_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.openwebtext_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.openwebtext_predict, batch_size=self.batch_size)

    def get_specs(self) -> dict[str, Any]:
        return {"batch_size": self.batch_size}
    