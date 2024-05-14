import json
from typing import Callable
from torch.utils.data import DataLoader
from datasets import DatasetDict
import datasets
from transformers import T5Tokenizer
from transformers import DataCollatorForSeq2Seq

from pytorch_fob.tasks import TaskDataModule
from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.engine.utils import log_info

MAX_TOKENS_PER_SENTENCE = 128

def generate_collate_fn_train(tokenizer, src_language: str, tgt_language: str) -> Callable:
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                      padding=True,
                                      label_pad_token_id=tokenizer.pad_token_id,
                                      return_tensors="pt")
    def collate(batch) -> object:
        res = []
        for sample in batch:
            res += [{"input_ids": sample[src_language],
                    "attention_mask": [1.0] * len(sample[src_language]),
                    "labels": sample[tgt_language]}]
        return collator(res)
    return collate

def generate_collate_fn_validtest(src_language: str, tgt_language: str) -> Callable:
    def collate(batch) -> tuple[list[str], list[str]]:
        src_text = []
        tgt_text = []
        for sample in batch:
            src_text += [sample["translation"][src_language]]
            tgt_text += [sample["translation"][tgt_language]]
        return src_text, tgt_text
    return collate


class WMTDataModule(TaskDataModule):
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.translation_direction: str = config.model.translation_direction.strip()
        if not (self.translation_direction == "de-en" or self.translation_direction == "en-de"):
            raise ValueError("translation direction needs to be de-en or en-de!")
        self.source_language: str = self.translation_direction.split("-")[0]
        self.target_language: str = self.translation_direction.split("-")[1]
        self.collate_fn_validtest = generate_collate_fn_validtest(self.source_language, self.target_language)
        self.processed_data_dir = self.data_dir / "processed"
        self.cache_dir = self.data_dir / "cache"
        self.prepare_workers = max(1, min(self.workers, 5))
        self.info_file = self.processed_data_dir / "info.json"
        if self.info_file.exists():
            with open(self.info_file, "r", encoding="utf8") as f:
                info = json.load(f)
                self.train_data_len = info["train_data_len"]
        else:
            self.train_data_len = 0
        self.tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small", cache_dir=str(self.cache_dir))

    def _get_dataset(self) -> DatasetDict:
        ds = datasets.load_dataset("wmt17",
                                   data_dir=str(self.data_dir),
                                   language_pair=("de", "en"),
                                   cache_dir=str(self.cache_dir), trust_remote_code=True,
                                   num_proc=self.prepare_workers,
                                   revision="e126783e58293db0be0c11dca4a4d2e6f4dcf0cd")
        return ds  # type: ignore

    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        self.data_dir.mkdir(exist_ok=True)
        if self.info_file.exists():
            log_info("wmt already preprocessed")
            return
        ds = self._get_dataset()
        log_info("tokenizing data...")
        def transform_text(data):
            de = [example["de"] for example in data["translation"]]
            en = [example["en"] for example in data["translation"]]
            return {"de": self.tokenizer(de)["input_ids"],
                    "en": self.tokenizer(en)["input_ids"]}
        ds = ds.map(transform_text, batched=True)
        ds["train"] = ds["train"].remove_columns("translation")
        log_info("filtering sentences with too many tokens...")
        ds = ds.filter(lambda data: len(data["de"]) <= MAX_TOKENS_PER_SENTENCE and
                                    len(data["en"]) <= MAX_TOKENS_PER_SENTENCE, num_proc=self.prepare_workers)
        log_info("saving dataset...")
        ds.save_to_disk(self.processed_data_dir)
        log_info("saving additional information...")
        with open(self.processed_data_dir / "info.json", "w", encoding="utf8") as f:
            json.dump({"max_tokens": MAX_TOKENS_PER_SENTENCE,
                       "train_data_len": len(ds["train"]),
                       "val_data_len": len(ds["validation"]),
                       "test_data_len": len(ds["test"])}, f, indent=4)
        log_info("wmt preprocessed")

    def setup(self, stage):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """
        ds = datasets.load_from_disk(str(self.processed_data_dir))

        data_collator = generate_collate_fn_train(self.tokenizer,
                                                  self.source_language,
                                                  self.target_language)

        self.collate_fn = data_collator

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.data_train = ds["train"]  # .select(range(200))
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
