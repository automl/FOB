import json
import numpy as np
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download
from transformers import SegformerImageProcessor
from torchvision.transforms import v2
from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.tasks import TaskDataModule


class SegmentationDataModule(TaskDataModule):
    """
    DataModule for SceneParse150 semantic segmentation task.
    Implementation inspired by 🤗 examples and tutorials:
    - https://github.com/huggingface/transformers/tree/main/examples/pytorch/semantic-segmentation
    - https://huggingface.co/blog/fine-tune-segformer
    """
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.revision = "ac1c0c0e23875e74cd77aca0fd725fd6a35c3667"
        image_processor = SegformerImageProcessor.from_pretrained(
            "nvidia/mit-b0",
            cache_dir=self.data_dir
        )
        # image_processor = SegformerImageProcessor.from_pretrained(
        #     "nvidia/segformer-b0-finetuned-ade-512-512",
        #     cache_dir=self.data_dir
        # )
        tgt_size = (image_processor.size["width"], image_processor.size["height"])
        tv_train_transforms = v2.Compose([
            v2.RandomResizedCrop(
                size=tgt_size,
                scale=(0.5, 2.0)  # as stated in paper
            ),
            v2.RandomHorizontalFlip()
        ])
        tv_val_transforms = v2.Identity()

        def trainval_transforms(tv_transforms):
            def transforms_fn(batch):
                images = []
                targets = []
                for image, target in zip(batch["image"], batch["annotation"]):
                    img, tgt = tv_transforms(image, target)
                    images.append(img)
                    targets.append(tgt)
                return image_processor(images, targets, do_reduce_labels=True)
            return transforms_fn

        self.train_transforms = trainval_transforms(tv_train_transforms)
        self.val_transforms = trainval_transforms(tv_val_transforms)
        id2label, label2id = self._get_label_dicts()
        self.id2label = id2label
        self.label2id = label2id
        self.num_labels = len(id2label)

    def prepare_data(self):
        self.data_dir.mkdir(exist_ok=True)
        load_dataset("scene_parse_150", cache_dir=str(self.data_dir), trust_remote_code=True, revision=self.revision)

    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """
        if stage == "fit":
            self.data_train = self._load_dataset("train")
            self.data_val = self._load_dataset("validation")
            self.data_train.set_transform(self.train_transforms)
            self.data_val.set_transform(self.val_transforms)
        if stage == "validate":
            self.data_val = self._load_dataset("validation")
            self.data_val.set_transform(self.val_transforms)
        if stage == "test":
            # no labels available for test split, so we use val
            self.data_test = self._load_dataset("validation")
            self.data_test.set_transform(self.val_transforms)
        if stage == "predict":
            self.data_predict = self._load_dataset("test")
            self.data_predict.set_transform(self.val_transforms)

    def _load_dataset(self, split: str) -> Dataset:
        ds = load_dataset(
            "scene_parse_150",
            cache_dir=str(self.data_dir),
            split=split,
            trust_remote_code=True,
            revision=self.revision
        )
        return self._remove_invalid_images(ds)  # type:ignore

    def _remove_invalid_images(self, dataset: Dataset) -> Dataset:
        return dataset.filter(lambda d: np.array(d["image"]).ndim >= 3)

    def _get_label_dicts(self) -> tuple[dict[int, str], dict[str, int]]:
        repo_id = "huggingface/label-files"
        filename = "ade20k-id2label.json"
        dl = hf_hub_download(repo_id, filename, repo_type="dataset", cache_dir=self.data_dir)
        with open(dl, "r", encoding="utf8") as f:
            id2label = json.load(f)
        id2label = {int(k): v for k, v in id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
        return id2label, label2id  # type:ignore
