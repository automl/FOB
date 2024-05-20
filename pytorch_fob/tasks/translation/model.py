import sys
from torch.nn import Module
from transformers import AutoModelForSeq2SeqLM, T5Config
from sacrebleu.metrics import BLEU
from sacrebleu.metrics.bleu import BLEUScore
from pytorch_fob.engine.parameter_groups import GroupedModel
from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.engine.utils import some, log_warn
from pytorch_fob.optimizers import Optimizer
from pytorch_fob.tasks import TaskModel
from pytorch_fob.tasks.translation.data \
    import WMTDataModule, MAX_TOKENS_PER_SENTENCE


class GroupedTransformer(GroupedModel):
    def __init__(self, model: Module, num_beams: int, length_penalty: float) -> None:
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        super().__init__(model)

    def generate(self, inputs: list[str], tokenizer, device, num_beams=None, length_penalty=None) -> list[str]:
        token_inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(device)
        num_beams = some(num_beams, default=self.num_beams)
        length_penalty = some(length_penalty, default=self.length_penalty)
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
        self.bleu = BLEU()
        self.metric_cache_pred: list[str] = []
        self.metric_cache_trues: list[str] = []
        if self.tokenizer is None:
            raise Exception("prepare dataset before running the model!")
        model_config = T5Config.from_pretrained("google-t5/t5-small", cache_dir=str(data_module.cache_dir))
        model = AutoModelForSeq2SeqLM.from_config(model_config)
        model = GroupedTransformer(model, config.model.num_beams, config.model.length_penalty)
        super().__init__(model, optimizer, config)

    def training_step(self, batch, _batch_idx):
        return self.compute_and_log_loss(batch, "train_loss")

    def compute_bleu(self, preds: list[str], target: list[str]) -> float:
        assert len(preds) == len(target)
        try:
            result : BLEUScore = self.bleu.corpus_score(hypotheses=[p.strip() for p in preds],
                                                        references=[[t.strip() for t in target]])
            return result.score
        except ZeroDivisionError:
            log_warn("Error: Bleu Score computing resulted in a ZeroDivisionError", file=sys.stderr)
            return 0.0

    def validation_step(self, batch, _batch_idx):
        src, tgt = batch
        batch = self.tokenizer(src, text_target=tgt, padding=True, return_tensors="pt").to(self.device)
        self.compute_and_log_loss(batch, "val_loss")
        self.metric_cache_trues += tgt
        self.metric_cache_pred += self.model.generate(src, self.tokenizer, self.device, num_beams=1, length_penalty=1.0)

    def test_step(self, batch, _batch_idx):
        src, tgt = batch
        batch = self.tokenizer(src, text_target=tgt, padding=True, return_tensors="pt").to(self.device)
        self.compute_and_log_loss(batch, "test_loss")
        self.metric_cache_trues += tgt
        self.metric_cache_pred += self.model.generate(src, self.tokenizer, self.device)

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
