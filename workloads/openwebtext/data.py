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
from datasets import load_dataset, load_from_disk, DownloadConfig  # huggingface datasets
from workloads import WorkloadDataModule
from bob.runtime import DatasetArgs

class OpenWebTextDataModule(WorkloadDataModule):
    def __init__(self, dataset_args: DatasetArgs):
        super().__init__(dataset_args)
        self.batch_size = 0  # TODO
        self.train_val_split = [1, 1]  # TODO
        self.seed = 42
        self.cache_dir = self.data_dir / "cache-hugging"
        self.data_dir = self.data_dir / "openwebtext"

        # TODO do we need to normalize?
        # meanOfOpenWebText = torch.tensor(0)
        # stdOfOpenWebText = torch.tensor(1)
        # self.transform  = transforms.Compose([transforms.ToTensor(),transforms.Normalize(meanOfOpenWebText, stdOfOpenWebText)])  
        
    def prepare_data(self):
        """download"""
        # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
        
        download_config = DownloadConfig(extract_compressed_file=True,
                                         cache_dir=self.cache_dir
                                         )

        dataset = load_dataset("openwebtext",
                               data_dir=self.data_dir,
                               download_config=download_config,
                               num_proc=self.workers)
        print("openwebtext: Finished load_dataset of this workload")

        dataset.save_to_disk(self.data_dir)
        loaded = load_from_disk(self.data_dir)
        print(loaded)

        # owt by default only contains the 'train' split, so create a test split
        # split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
        # split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """
        dataset = load_dataset("openwebtext",
                               data_dir=self.data_dir,
                               num_proc=self.workers)
        split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
        split_dataset['val'] = split_dataset.pop('test') # rename the test split to val
        if stage == "fit":
            pass
        if stage == "test":
            pass
        if stage == "predict":
            pass

    def get_specs(self) -> dict[str, Any]:
        return {"batch_size": self.batch_size}
    