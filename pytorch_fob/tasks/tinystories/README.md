# Task

This task does language modelling with a small GPT model. It is trained to predict the next token given the previous tokens.

## Dataset

We use the [tiny stories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset proposed in [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759). It contains synthetically generated (by GPT-3.5 and GPT-4) short stories that only use a small vocabulary.

## Model

We use a small version of the [nanoGPT](https://github.com/karpathy/nanoGPT) model with only 8M parameters.

## Performance

The metric measured here is the perplexity. We expect a performance of `7.58`.
