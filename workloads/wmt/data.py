from typing import Any, Iterable, Iterator
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from workloads import WorkloadDataModule
from runtime import DatasetArgs
import datasets
from datasets import DatasetDict, Dataset
import os
from tqdm import tqdm
import json
import pickle

# code inspired by: https://pytorch.org/tutorials/beginner/translation_transformer.html
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
MAX_TOKENS_PER_SENTENCE = 256
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func


def tensor_transform(token_ids: list[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


class WMTDataModule(WorkloadDataModule):
    def __init__(self, runtime_args: DatasetArgs):
        super().__init__(runtime_args)
        self.data_dir = self.data_dir / "wmt"
        self.processed_data_dir = self.data_dir / "processed"
        self.cache_dir = self.data_dir / "cache"
        self.prepare_workers = max(1, min(self.workers, 5))
        self.batch_size = 128
        self.tokenizer = {}
        self.vocab_transform = {}
        self.info_file = self.processed_data_dir / "info.json"
        if self.info_file.exists():
            with open(self.info_file, "r", encoding="utf8") as f:
                info = json.load(f)
                self.vocab_size = info["vocab_size"]
                self.train_data_len = info["train_data_len"]
        else:
            self.vocab_size: dict[str, int] = {}
            self.train_data_len = 0

    def _get_dataset(self) -> DatasetDict:
        ds = datasets.load_dataset("wmt17",
                                   data_dir=str(self.data_dir),
                                   language_pair=("de", "en"),
                                   cache_dir=str(self.cache_dir), trust_remote_code=True,
                                   num_proc=self.prepare_workers)
        return ds  # type: ignore

    def _load_tokenizer(self):
        if len(self.tokenizer) < 2:
            self.tokenizer["de"] = get_tokenizer('spacy', language='de_core_news_sm')
            self.tokenizer["en"] = get_tokenizer('spacy', language='en_core_web_sm')

    def _prepare_data_transform(self, train_data: Dataset | None = None):
        self._load_tokenizer()
        if (self.processed_data_dir / "vocabulary.pkl").exists():
            print("loading vocabulary from disk...")
            with open(self.processed_data_dir / "vocabulary.pkl", "rb") as f:
                for k, v in pickle.load(f).items():
                    self.vocab_transform[k] = v
                for ln in ("de", "en"):
                    self.vocab_size[ln] = len(self.vocab_transform[ln])
        elif train_data is not None:
            for ln in ("de", "en"):
                print(f"building vocabulary for {ln}...")
                self.vocab_transform[ln] = build_vocab_from_iterator(
                    self._yield_token(train_data, ln),
                    min_freq=1,
                    specials=special_symbols,
                    special_first=True,
                    max_tokens=32_000
                )
                self.vocab_transform[ln].set_default_index(UNK_IDX)
                self.vocab_size[ln] = len(self.vocab_transform[ln])
        else:
            raise Exception("Vocabulary not found and it should not be created!")

    def _yield_token(self, data_iter: Iterable, language: str) -> Iterator[str]:
        for data in tqdm(data_iter):
            yield self.tokenizer[language](data["translation"][language])

    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        self.data_dir.mkdir(exist_ok=True)
        if self.info_file.exists():
            print("wmt already preprocessed")
            return
        ds = self._get_dataset()
        exit = os.system("python -m spacy download en_core_web_sm")
        if exit != 0:
            raise Exception("en tokenizer download failed")
        exit = os.system("python -m spacy download de_core_news_sm")
        if exit != 0:
            raise Exception("de tokenizer download failed")
        print("preparing data...")
        self._prepare_data_transform(ds["train"])
        print("transforming data...")
        def transform_text(data):
            res = {}
            for ln in ("de", "en"):
                res[ln] = sequential_transforms(
                    self.tokenizer[ln],
                    self.vocab_transform[ln],
                    tensor_transform)(data["translation"][ln])
            return res
        ds = ds.map(transform_text, num_proc=self.prepare_workers)
        ds["train"] = ds["train"].remove_columns("translation")

        print("filtering sentences with too many tokens...")
        ds = ds.filter(lambda data: len(data["de"]) <= MAX_TOKENS_PER_SENTENCE
                       and len(data["en"]) <= MAX_TOKENS_PER_SENTENCE, num_proc=self.prepare_workers)

        print("saving dataset...")
        ds.save_to_disk(self.processed_data_dir)
        print("saving vocabulary...")
        with open(self.processed_data_dir / "vocabulary.pkl", "wb") as f:
            pickle.dump(self.vocab_transform, f)
        print("saving additional information...")
        with open(self.processed_data_dir / "info.json", "w", encoding="utf8") as f:
            json.dump({"vocab_size": self.vocab_size,
                       "max_tokens": MAX_TOKENS_PER_SENTENCE,
                       "train_data_len": len(ds["train"]),
                       "val_data_len": len(ds["validation"]),
                       "test_data_len": len(ds["test"])}, f, indent=4)
        print("wmt preprocessed")

    def setup(self, stage):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """
        ds = datasets.load_from_disk(str(self.processed_data_dir))
        self._prepare_data_transform()

        def collate_fn(batch):
            src_batch, tgt_batch = [], []
            for sample in batch:
                src_batch.append(torch.tensor(sample["de"]))
                tgt_batch.append(torch.tensor(sample["en"]))

            src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
            tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
            return src_batch, tgt_batch
        
        self.collate_fn = collate_fn

        def collate_fn_validtest(batch):
            de_text = []
            en_text = []
            for sample in batch:
                de_text += [sample["translation"]["de"]]
                en_text += [sample["translation"]["en"]]
            return collate_fn(batch) + (de_text, en_text)

        self.collate_fn_validtest = collate_fn_validtest

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.data_train = ds["train"]
            self.data_val = ds["validation"]

        if stage == "validate":
            self.data_val = ds["validation"]
        
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.data_test = ds["test"]

        if stage == "predict":
            self.data_predict = ds["test"]

    def train_dataloader(self):
        self.check_dataset(self.data_train)
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        self.check_dataset(self.data_val)
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn_validtest
        )

    def test_dataloader(self):
        self.check_dataset(self.data_test)
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn_validtest
        )

    def predict_dataloader(self):
        self.check_dataset(self.data_predict)
        return DataLoader(
            self.data_predict,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn_validtest
        )
