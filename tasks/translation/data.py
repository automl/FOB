import json
from torch.utils.data import DataLoader
from datasets import DatasetDict
import datasets
from transformers import T5Tokenizer
from transformers import DataCollatorForSeq2Seq

from tasks import TaskDataModule
from engine.configs import TaskConfig

MAX_TOKENS_PER_SENTENCE = 128


def collate_fn_validtest(batch) -> tuple[list[str], list[str]]:
    de_text = []
    en_text = []
    for sample in batch:
        de_text += [sample["translation"]["de"]]
        en_text += [sample["translation"]["en"]]
    return de_text, en_text


class WMTDataModule(TaskDataModule):
    def __init__(self, config: TaskConfig):
        super().__init__(config)
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
        self.tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base", cache_dir=str(self.cache_dir))

    def _get_dataset(self) -> DatasetDict:
        ds = datasets.load_dataset("wmt17",
                                   data_dir=str(self.data_dir),
                                   language_pair=("de", "en"),
                                   cache_dir=str(self.cache_dir), trust_remote_code=True,
                                   num_proc=self.prepare_workers)
        return ds  # type: ignore

    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        self.data_dir.mkdir(exist_ok=True)
        if self.info_file.exists():
            print("wmt already preprocessed")
            return
        ds = self._get_dataset()
        print("tokenizing data...")
        def transform_text(data):
            inputs = [example["de"] for example in data["translation"]]
            targets = [example["en"] for example in data["translation"]]
            return self.tokenizer(inputs, text_target=targets)
        ds = ds.map(transform_text, batched=True)
        ds["train"] = ds["train"].remove_columns("translation")
        print("filtering sentences with too many tokens...")
        ds = ds.filter(lambda data: len(data["input_ids"]) <= MAX_TOKENS_PER_SENTENCE and
                                    len(data["labels"]) <= MAX_TOKENS_PER_SENTENCE, num_proc=self.prepare_workers)
        print("saving dataset...")
        ds.save_to_disk(self.processed_data_dir)
        print("saving additional information...")
        with open(self.processed_data_dir / "info.json", "w", encoding="utf8") as f:
            json.dump({"max_tokens": MAX_TOKENS_PER_SENTENCE,
                       "train_data_len": len(ds["train"]),
                       "val_data_len": len(ds["validation"]),
                       "test_data_len": len(ds["test"])}, f, indent=4)
        print("wmt preprocessed")

    def setup(self, stage):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """
        ds = datasets.load_from_disk(str(self.processed_data_dir))

        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer,
                                               padding=True,
                                               label_pad_token_id=self.tokenizer.pad_token_id,
                                               return_tensors="pt")

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
            collate_fn=collate_fn_validtest
        )

    def test_dataloader(self):
        self.check_dataset(self.data_test)
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=collate_fn_validtest
        )

    def predict_dataloader(self):
        self.check_dataset(self.data_predict)
        return DataLoader(
            self.data_predict,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=collate_fn_validtest
        )
