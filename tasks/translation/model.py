import sys
import math
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import evaluate

from transformers import BertModel, BertConfig, AutoModelForSeq2SeqLM, PretrainedConfig, EncoderDecoderConfig, EncoderDecoderModel, AutoConfig, AutoModel, T5Config

from engine.parameter_groups import GroupedModel, ParameterGroup, merge_parameter_splits
from engine.configs import TaskConfig
from optimizers import Optimizer
from tasks import TaskModel
from tasks.translation.data \
    import WMTDataModule, MAX_TOKENS_PER_SENTENCE


class GroupedTransformer(GroupedModel):
    def __init__(self, model) -> None:
        super().__init__(model)

    def generate(self, inputs: list[str], tokenizer, device) -> list[str]:
        inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(device)
        output = self.model.generate(input_ids=inputs["input_ids"],
                                     attention_mask=inputs["attention_mask"],
                                     do_sample=False,
                                     max_length=MAX_TOKENS_PER_SENTENCE - 2,
                                     num_beams=4,
                                     length_penalty=0.6)
        return tokenizer.batch_decode(output, skip_special_tokens=True)

    def parameter_groups(self) -> list[ParameterGroup]:
        split1 = super().parameter_groups()  # default split
        # split2 = [ParameterGroup(dict(self.model.named_parameters()), lr_multiplier=0.1)]  # use less learning rate
        return split1  # merge_parameter_splits(split1, split2)


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
        # encoder_config = BertConfig(vocab_size=self.tokenizer.vocab_size,
        #                             bos_token_id=self.tokenizer.bos_token_id,
        #                             eos_token_id=self.tokenizer.eos_token_id,
        #                             unk_token_id=self.tokenizer.unk_token_id,
        #                             pad_token_id=self.tokenizer.pad_token_id)
        # decoder_config = BertConfig(vocab_size=self.tokenizer.vocab_size,
        #                             bos_token_id=self.tokenizer.bos_token_id,
        #                             eos_token_id=self.tokenizer.eos_token_id,
        #                             unk_token_id=self.tokenizer.unk_token_id,
        #                             pad_token_id=self.tokenizer.pad_token_id,
        #                             is_decoder=True,
        #                             add_cross_attention=True)
        # config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config,
        #                                                            bos_token_id=self.tokenizer.bos_token_id,
        #                                                            eos_token_id=self.tokenizer.eos_token_id,
        #                                                            unk_token_id=self.tokenizer.unk_token_id,
        #                                                            pad_token_id=self.tokenizer.pad_token_id)
        # model = EncoderDecoderModel(config=config)
        model_config = T5Config.from_pretrained("google-t5/t5-base", cache_dir=str(data_module.cache_dir))
        model = AutoModelForSeq2SeqLM.from_config(model_config)
        model = GroupedTransformer(model)
        super().__init__(model, optimizer, config)

    def training_step(self, batch, batch_idx):
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

    def validation_step(self, batch, batch_idx):
        de, en = batch
        batch = self.tokenizer(de, text_target=en, padding=True, return_tensors="pt").to(self.device)
        self.compute_and_log_loss(batch, "val_loss")
        self.metric_cache_trues += en
        self.metric_cache_pred += self.model.generate(de, self.tokenizer, self.device)

    def test_step(self, batch, batch_idx):
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
