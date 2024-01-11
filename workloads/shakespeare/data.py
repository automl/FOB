from typing import Any
from pathlib import Path
import torch
import os
import pickle
import requests
import numpy as np
from tqdm import tqdm


from workloads import WorkloadDataModule
from runtime import DatasetArgs

class ShakespeareDataModule(WorkloadDataModule):
    def __init__(self, dataset_args: DatasetArgs):
        super().__init__(dataset_args)
        self.data_dir = self.data_dir / "tinyshakespeare"
        self.seed = 42
        self.shuffle = True
        self.batch_size = 64

        # TODO do we need to normalize?
        # meanOfOpenWebText = torch.tensor(0)
        # stdOfOpenWebText = torch.tensor(1)
        # self.transform  = transforms.Compose([transforms.ToTensor(),transforms.Normalize(meanOfOpenWebText, stdOfOpenWebText)])  


    def prepare_data(self):
        """thanks to https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py"""
        print("tinyshakespeare: Begin download of this workload")
        input_file_path = self.data_dir / 'input.txt'
        if not os.path.exists(input_file_path):
            # make sure folder exists
            Path(self.data_dir).mkdir(parents=False, exist_ok=True) 
            data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            with open(input_file_path, 'w') as f:
                f.write(requests.get(data_url).text)

        with open(input_file_path, 'r') as f:
            data = f.read()
        print(f"length of dataset in characters: {len(data):,}")

        # get all the unique characters that occur in this text
        chars = sorted(list(set(data)))
        vocab_size = len(chars)
        print("all the unique characters:", ''.join(chars))
        print(f"vocab size: {vocab_size:,}")

        # create a mapping from characters to integers
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        def encode(s):
            return [stoi[c] for c in s] # encoder: take a string, output a list of integers
        def decode(l):
            return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

        # create the train and test splits
        n = len(data)
        train_data = data[:int(n*0.9)]
        val_data = data[int(n*0.9):]

        # encode both to integers
        train_ids = encode(train_data)
        val_ids = encode(val_data)
        print(f"train has {len(train_ids):,} tokens")
        print(f"val has {len(val_ids):,} tokens")

        # export to bin files
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)
        train_ids.tofile(self.data_dir / 'train.bin')
        val_ids.tofile(self.data_dir / 'val.bin')

        # save the meta information as well, to help us encode/decode later
        meta = {
            'vocab_size': vocab_size,
            'itos': itos,
            'stoi': stoi,
        }
        print("tinyshakespeare: save tokenized data to disk")
        with open(self.data_dir / 'meta.pkl', 'wb') as f:
            pickle.dump(meta, f)

    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """
        # print(f"Load from cache at {str(self.data_dir)}")
        # dataset = load_from_disk(self.data_dir)

        dataset = self._load_tokenized_from_disk()
        print("tinyshakespear: Begin splitting the dataset")
        # owt by default only contains the 'train' split, so create a test split and rename it to val

        # TODO split in 3 parts
        split_dataset = dataset["train"].train_test_split(test_size=self.test_size,
                                                          seed=self.seed,
                                                          shuffle=self.shuffle)
        split_dataset['val'] = split_dataset.pop('test')

        self.dataset_train = split_dataset["train"]
        self.dataset_val = split_dataset["val"]
        # self.dataset_test = split_dataset["test"]  # TODO
        
        
        if stage == "fit":
            return self.train_dataloader()
        if stage == "test":
            return self.val_dataloader()
        if stage == "predict":
            return self.test_dataloader()

    def get_specs(self) -> dict[str, Any]:
        return {"batch_size": self.batch_size}
    