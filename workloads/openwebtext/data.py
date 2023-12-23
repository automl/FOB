# https://huggingface.co/datasets/Skylion007/openwebtext

# maybe we can joink code from here:
# https://github.com/karpathy/nanoGPT/tree/master/data/openwebtext
# that guy joinked it from here: (also lightning)
# https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

# if not joinkable; have a look here
# https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/builder_classes#datasets.DownloadManager
# downloaded_files = dl_manager.download('https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz')
# extracted_files = dl_manager.extract(downloaded_files)

import torch
import os
import numpy as np
import tqdm
from transformers import AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk, DownloadConfig  # huggingface datasets
from workloads import WorkloadDataModule
from bob.runtime import DatasetArgs

class OpenWebTextDataModule(WorkloadDataModule):
    def __init__(self, dataset_args: DatasetArgs):
        super().__init__(dataset_args)
        self.cache_dir = self.data_dir / "cache-hugging"
        #self.data_dir = self.data_dir / "openwebtext"
        self.data_dir = self.data_dir / "rotten-test-token"
        # self.train_val_split = [1, 1]  # TODO use test_size atm
        self.test_size = 0.0005
        self.seed = 42
        self.shuffle = True
        self.batch_size = 0  # TODO

        self.tokenizer = self.prepare_tokenizer()
        self.prepare_pretrained_model()

        # TODO do we need to normalize?
        # meanOfOpenWebText = torch.tensor(0)
        # stdOfOpenWebText = torch.tensor(1)
        # self.transform  = transforms.Compose([transforms.ToTensor(),transforms.Normalize(meanOfOpenWebText, stdOfOpenWebText)])  
    
    def prepare_tokenizer(self):
        # GPT-2 is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left.
        # tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=self.cache_dir)
        return tokenizer
    
    def prepare_pretrained_model(self):
        """TODO download and cache pretrained model weights here?"""
        model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=self.cache_dir)


    def prepare_data(self):
        """takes 54GB in huggingface cache dir, about 8M documents (8,013,769)"""
        download_config = DownloadConfig(extract_compressed_file=True,
                                         cache_dir=self.cache_dir
                                         )
        print("openwebtext: Begin load_dataset of this workload")
        #dataset = load_dataset("openwebtext",
        dataset = load_dataset("rotten_tomatoes",
                               data_dir=self.data_dir,
                               download_config=download_config,
                               num_proc=self.workers)
        print(dataset)

        # TODO: we could also just reuse the data from the cache directory (instead of saving it here), but i like to have all data at the same place
        #       definitely think about integrating the huggingface cache if we use more of those datasets

        # TODO we save tokenized data instead
        # dataset.save_to_disk(self.data_dir)

        # The tokenizer returns a dictionary with three items:
            # input_ids: the numbers representing the tokens in the text.
            # token_type_ids: indicates which sequence a token belongs to if there is more than one sequence.
            # attention_mask: indicates whether a token should be masked or not.
        
        def tokenizing(example):
            return self.tokenizer(example["text"],
                                  max_length=1024,
                                  truncation=True,
                                  return_tensors="pt")

        print("openwebtext: Begin tokenizing splits")
        tokenized = dataset.map(tokenizing,
                                remove_columns=['text'],  # TODO is this ok?
                                # batched=True,  # TODO check if this is good and how to make it work
                                desc="tokenizing OpenWebText splits",
                                num_proc=self.workers
                                )
        print("openwebtext: save tokenized data to disk")
        self._save_tokenized_to_disk(tokenized)
        
        
    def _save_tokenized_to_disk(self, tokenized_data):
        # slightly adapted from https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py
        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized_data.items():
            # Calculate total number of tokens # TODO cleaner
            print(tokenized_data.column_names)
            arr_len = np.sum([len(ids) for ids in tokenized_data["input_ids"]])
            # arr_len = np.sum(dset['len'], dtype=np.uint64)
            
            filename = self.data_dir / f"{split}.bin"
            dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            total_batches = 1024

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                # Batch together samples for faster write
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()
    
    def _load_tokenized_from_disk(self, set: str):
        assert set in ["train"]
        filename = self.data_dir / f"{set}.bin"
        return np.memmap(filename, dtype=np.uint16, mode='r')

    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """
        # print(f"Load from cache at {str(self.data_dir)}")
        # dataset = load_from_disk(self.data_dir)

        dataset = self._load_tokenized_from_disk()
        print("openwebtext: Begin splitting the dataset")
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
