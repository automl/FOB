# Task

A translation task between German and English sentences which
are a part of the WMT17 dataset.
We can train the model for both translation directions individually.
The German to English translation direction is a bit easier than English to German.

## Dataset

The dataset comes from the 2017 machine translation task [WMT17](https://www.statmt.org/wmt17/translation-task.html).
Here we use the subset of german-english sentences and filter out sentences which
would lead to more than 128 tokens. This results in 5864155 training sentences.
The test set consists of 3003 tokens and is the same as the WMT14 test set.

## Model

The task uses the T5-small model and corresponding tokenizer from the paper
[Exploring the Limits of Transfer Learning with a Unified
Text-to-Text Transformer](https://jmlr.org/papers/volume21/20-074/20-074.pdf)
with the default settings. This results in 60.5M parameters
which is a relatively small model compared models which are normally used in the competition.

## Performance

When running large transformer models the Adafactor optimizer is used quite often to reduce the memory footprint of AdamW. To enable a comparison between optimizers
a batch size is used which is possible to run with AdamW as well.
For the baseline score following grid search was used: [experiment](../../baselines/translation.yaml)

- 31.8 BLEU for de-en with AdamW and learning rate 1e-3, weight decay 0.1
- 26.3 BLEU for en-de with AdamW and learning rate 1e-3, weight decay 0.1

### Performance Comparison

We use beam search with 4 beams and length penalty 0.6.
The BLEU score is computed using sacreBLEU
(there are other BLEU implementations but they yield different results).
Those settings were used in the two comparison papers as well.

As comparison [AlgoPerf](https://arxiv.org/abs/2306.07179) uses a transformer model with ~133.5M parameters which is ~2x the size. On WMT17 it achieves with
the same batch size and similar training time ~31 BLEU on de-en.

The [paper](https://jmlr.org/papers/volume21/20-074/20-074.pdf) in which the T5 model was introduced, they achieve 26.7 BLEU on en-de. The comparison is a bit difficult because they use WMT16 which only has ~4.5M sentences instead of ~5.9M and they pre-train the model on the C4 dataset. In favor of training time we do not pre-train the model and still achieve an acceptable performance.

