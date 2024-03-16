import sys
import evaluate

from transformers import AutoModelForSeq2SeqLM, T5Config

from engine.parameter_groups import GroupedModel
from engine.configs import TaskConfig
from optimizers import Optimizer
from tasks import TaskModel
from tasks.translation.data \
    import WMTDataModule, MAX_TOKENS_PER_SENTENCE


class GroupedTransformer(GroupedModel):
    def generate(self, inputs: list[str], tokenizer, device, num_beams=4, length_penalty=1.0) -> list[str]:
        token_inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(device)
        output = self.model.generate(input_ids=token_inputs["input_ids"],
                                     attention_mask=token_inputs["attention_mask"],
                                     do_sample=False,
                                     max_length=MAX_TOKENS_PER_SENTENCE - 2,
                                     num_beams=num_beams,
                                     length_penalty=length_penalty)
        return tokenizer.batch_decode(output, skip_special_tokens=True)


class WMTModel(TaskModel):
    def __init__(self, optimizer: Optimizer, data_module: WMTDataModule, config: TaskConfig):
        self.batch_size = data_module.batch_size
        self.train_data_len = data_module.train_data_len
        self.tokenizer = data_module.tokenizer
        self.bleu = evaluate.load("sacrebleu", cache_dir=str(data_module.cache_dir))
        self.metric_cache_pred: list[str] = []
        self.metric_cache_trues: list[str] = []
        if self.tokenizer is None:
            raise Exception("prepare dataset before running the model!")
        model_config = T5Config.from_pretrained("google-t5/t5-small", cache_dir=str(data_module.cache_dir))
        model = AutoModelForSeq2SeqLM.from_config(model_config)
        model = GroupedTransformer(model)
        super().__init__(model, optimizer, config)

    def training_step(self, batch, _batch_idx):
        return self.compute_and_log_loss(batch, "train_loss")

    def compute_bleu(self, en_preds: list[str], en_target: list[str]) -> float:
        assert len(en_preds) == len(en_target)
        try:
            result = self.bleu.compute(predictions=[p.strip() for p in en_preds],
                                       references=[[t.strip()] for t in en_target])
        except ZeroDivisionError:
            print("Error: Bleu Score computing resulted in a ZeroDivisionError", file=sys.stderr)
            result = {"score": 0.0}
        return result["score"]  # type: ignore

    def validation_step(self, batch, _batch_idx):
        de, en = batch
        batch = self.tokenizer(de, text_target=en, padding=True, return_tensors="pt").to(self.device)
        self.compute_and_log_loss(batch, "val_loss")
        self.metric_cache_trues += en
        self.metric_cache_pred += self.model.generate(de, self.tokenizer, self.device, num_beams=1, length_penalty=1.0)

    def test_step(self, batch, _batch_idx):
        de, en = batch
        batch = self.tokenizer(de, text_target=en, padding=True, return_tensors="pt").to(self.device)
        self.compute_and_log_loss(batch, "test_loss")
        self.metric_cache_trues += en
        self.metric_cache_pred += self.model.generate(de, self.tokenizer, self.device)


    def on_validation_epoch_end(self):
        bleu = self.compute_bleu(self.metric_cache_pred, self.metric_cache_trues)
        self.log("val_bleu", bleu, batch_size=self.batch_size, sync_dist=True)
        self.metric_cache_trues.clear()
        self.metric_cache_pred.clear()

    def on_test_epoch_end(self) -> None:
        bleu = self.compute_bleu(self.metric_cache_pred, self.metric_cache_trues)
        self.log("test_bleu", bleu, batch_size=self.batch_size, sync_dist=True)
        self.metric_cache_trues.clear()
        self.metric_cache_pred.clear()

    def compute_and_log_loss(self, batch, log_name: str):
        labels = batch.labels
        # replace padding token id's of the labels by -100 so it's ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100
        output = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=labels)
        self.log(log_name, output.loss, batch_size=self.batch_size, sync_dist=True)
        return output.loss
