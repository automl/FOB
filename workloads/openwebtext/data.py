# https://huggingface.co/datasets/Skylion007/openwebtext

# maybe we can joink code from here:
# https://github.com/karpathy/nanoGPT/tree/master/data/openwebtext
# that guy joinked it from here: (also lightning)
# https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

# if not joinkable; have a look here
# https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/builder_classes#datasets.DownloadManager
# downloaded_files = dl_manager.download('https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz')
# extracted_files = dl_manager.extract(downloaded_files)

from typing import Any
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset # huggingface datasets
from workloads import WorkloadDataModule

class OpenWebTextDataModule(WorkloadDataModule):
    def __init__(self, data_dir: Path):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = 0  # TODO
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

    def get_specs(self) -> dict[str, Any]:
        return {"batch_size": self.batch_size}
    