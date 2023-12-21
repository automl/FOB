from typing import Any
from pathlib import Path
import torch
from torch.utils.data import DataLoader
# from torchvision import transforms
from workloads import WorkloadDataModule
from bob.runtime import DatasetArgs

class OGBGDataModule(WorkloadDataModule):
    def __init__(self, runtime_args: DatasetArgs):
        super().__init__(runtime_args)
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

    def get_specs(self) -> dict[str, Any]:
        return {"batch_size": self.batch_size}
    