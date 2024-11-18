import json

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from tiktoken import get_encoding
from torch.utils.data import DataLoader, Dataset, RandomSampler

from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.engine.utils import log_info
from pytorch_fob.tasks import TaskDataModule


class TinyStoriesDataset(Dataset):
    def __init__(self, tokens: list[int], block_size):
        self.block_size = block_size
        self.tokens = tokens

    def __len__(self):
        # Return number of possible sequences
        # -1 because we need room for the target (shifted by 1)
        return len(self.tokens) - self.block_size - 1

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx : idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1 : idx + self.block_size + 1], dtype=torch.long)
        return x, y


class TinyStoriesDataModule(TaskDataModule):
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.block_size = config.model.block_size
        self.samples_per_epoch = config.samples_per_epoch
        self.tokenizer = get_encoding("gpt2")
        self.processed_data_dir = self.data_dir / "processed"
        self.cache_dir = self.data_dir / "cache"

    def prepare_data(self) -> None:
        if self.processed_data_dir.exists():
            return

        log_info("loading dataset...")
        ds: DatasetDict = load_dataset(
            "roneneldan/TinyStories",
            cache_dir=str(self.cache_dir),
            trust_remote_code=True,
            revision="42639e24f1c4e7bb6d432ea2a93946bca7cb16e1",
        )  # type: ignore

        def tokenize_function(batch: dict[str, list]) -> dict[str, list]:
            tokens = []
            for text in batch["text"]:
                if len(text.strip()) > 0:  # Skip empty texts
                    tokens.extend(self.tokenizer.encode_ordinary(text))
                    tokens.append(self.tokenizer.eot_token)  # Add end of text token between texts
            return {"tokens": tokens}

        log_info("tokenizing data...")
        ds = ds.map(function=tokenize_function, batched=True, remove_columns=["text"])

        log_info("saving processed dataset...")
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(self.processed_data_dir)

        log_info("saving additional information...")
        info = {
            "vocab_size": self.tokenizer.n_vocab,
            "train_data_len": len(ds["train"]),
            "val_data_len": len(ds["validation"]),
        }
        with open(self.processed_data_dir / "info.json", "w", encoding="utf8") as f:
            json.dump(info, f, indent=4)
        log_info("TinyStories preprocessed")

    def setup(self, stage):
        log_info("loading dataset...")
        if stage == "fit":
            ds: DatasetDict = load_from_disk(str(self.processed_data_dir))  # type: ignore
            self.data_train = self._wrap_dataset(ds["train"])
            self.data_val = self._wrap_dataset(ds["validation"])
        else:
            ds: Dataset = load_from_disk(str(self.processed_data_dir / "validation"))  # type: ignore
            if stage == "validate":
                self.data_val = self._wrap_dataset(ds)
            if stage == "test":
                self.data_test = self._wrap_dataset(ds)
            if stage == "predict":
                self.data_predict = self._wrap_dataset(ds)

    def train_dataloader(self):
        rs = RandomSampler(
            self.data_train,
            num_samples=self.train_samples,
            replacement=True,
        )
        dl = DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
            sampler=rs,
        )
        return dl

    def val_dataloader(self):
        rs = RandomSampler(
            self.data_val,
            num_samples=self.val_samples,
            replacement=True,
        )
        dl = DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
            sampler=rs,
        )
        return dl

    def test_dataloader(self):
        sampler = RandomSampler(
            self.data_test,
            num_samples=self.test_samples,
            replacement=False,
            generator=torch.Generator().manual_seed(42),
        )
        dl = DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
            sampler=sampler,
        )
        return dl

    @property
    def train_samples(self):
        return self.samples_per_epoch.train

    @property
    def val_samples(self):
        return self.samples_per_epoch.val

    @property
    def test_samples(self):
        return self.samples_per_epoch.test

    def get_vocab_size(self) -> int:
        return self.tokenizer.n_vocab

    def _wrap_dataset(self, dataset: Dataset) -> TinyStoriesDataset:
        return TinyStoriesDataset(dataset["tokens"], self.block_size)
