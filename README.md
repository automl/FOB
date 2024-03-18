# Fast Optimization Benchmark

Fast and cheap Benchmark for HPO and Optimizer.

Master Project at ML chair Freiburg,
Simon Blauth, Tobias Bürger, Zacharias Häringer

This benchmark aims to be fast while maintaining a wide selection of different tasks. It also tries to be independent of the hardware used, however it requires a minimum of 4 gpus ideally capable of bfloat16 mixed precision.  
One run of all tasks in this suite on a single optimizer configuration should not take more than a day.
A benchmark should state the following for each task: time taken per optimization step compared to baseline, best model performance, final model performance. 

## Tasks

We try to cover a large range of deep learning tasks in this benchmark.  
Some of these are still under development.

How to write your own can be found [here](tasks/README.md)

### Ready to use

| Name | Dataset | Model | Task | Target Metric | Baseline Score | Baseline Runtime | Hardware |
| ------- | ---- | ----- | ---- | ------------- | -------------- | ---------------- | -------- |
| mnist | MNIST | MLP | Image classification | Top-1 Accuracy | 0.97 | 1 min | 1 gpu |
| classification | [Imagenet-64x64](https://patrykchrabaszcz.github.io/Imagenet32/) | [Wide ResNet](https://arxiv.org/pdf/1605.07146.pdf) | Image classification | Top-1 Accuracy | 0.69 | 4h | 4 gpu |
| classification_small | [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) | [Resnet18](https://arxiv.org/pdf/1512.03385.pdf) | Image classification | Top-1 Accuracy | 0.77 | 10 min | 1 gpu |
| segmentation | [MIT Scene Parse](http://sceneparsing.csail.mit.edu/) | [SegFormer](https://arxiv.org/abs/2105.15203) | Semantic Segmentation | Intersection over Union (IoU) | 0.35 | 5h | 4 gpu |
| graph_tiny | [cora](https://paperswithcode.com/sota/node-classification-on-cora) | [GCN](https://arxiv.org/abs/1609.02907) | Node Classification | Accuracy | 0.80 | 1min | 1 gpu |
| tabular | [California Housing](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html) | [FT Transformer](https://arxiv.org/pdf/2106.11959.pdf) | tabular regression | Test MSE | 0.11 | 2 min | 1 gpu |
| translation | [WMT17(de-en)](https://machinetranslate.org/wmt17) (maybe subset in the future) | [T5 small](https://jmlr.org/papers/volume21/20-074/20-074.pdf) | machine translation | BLEU (sacrebleu) | 31 | 9h | 4 gpus |


### Under Development

| Name | Dataset | Model | Task | Target Metric | Baseline Score | Baseline Runtime | Hardware |
| ------- | ----- | ----- | ---- | ------------- | -------------- | ---------------- | -------- |
| graph | [ogbg-molhiv](https://ogb.stanford.edu/docs/graphprop/#ogbg-mol) | [Graph Isomorphism Network (GIN)](https://arxiv.org/pdf/1810.00826.pdf) | graph property prediction | ROC-AUC | 0.73? | 20min | 1 gpu |
| detection | [COCO](https://cocodataset.org) | [Faster R-CNN](https://arxiv.org/abs/1506.01497) with [MobileNet v3](https://arxiv.org/abs/1905.02244) backbone | Object detection | Average Precision (IoU) | ? | ~4h | 4 gpus |
| speech | librispeech | conformer | speech recognition | ? | ? | ? | ? |
| rna_folding | bpRNA | RNAformer | RNA secondary structure prediction | F1 | ? | ~4h | 4 gpus |
| diffusion | FFHQ | ? | image diffusion | ? | ? | ? | ? |



## Optimizer and Scheduler

An optimizer (together with the scheduler) contains the deep learning training algorithm to benchmark. Each optimizer has its own subfolder in the `optimizers` folder.
We currently have the following optimizers:

| Name | Optimizer | LR Scheduler |
| ---- | --------- | ------------ |
| adamw_baseline | [AdamW](https://arxiv.org/abs/1711.05101) | [Cosine Annealing](https://arxiv.org/abs/1608.03983) with linear warmup |
| adamcpr | [AdamCPR](https://arxiv.org/abs/2311.09058v2) | [Cosine Annealing](https://arxiv.org/abs/1608.03983) with linear warmup |
| sgd_baseline | Stochastic Gradient Descent | [Cosine Annealing](https://arxiv.org/abs/1608.03983) |

How to write your own can be found [here](optimizers/README.md)

## Usage Instructions

### Installation

This repo was tested with python 3.10, but any version >= 3.10 should work.  
Create conda environment:
```
conda create -n fob python=3.10 -y
```
Activate and install requirements
```
conda activate fob
pip install -r requirements.txt
```

### How to run an experiment

Make sure you have the conda environment set up and activated.
Then you write an `experiment.yaml` (can be named differently) where you specify which optimizer and task you want to use. Every value can also be a list of values if you want to perform a gridsearch over them (more details below).

As an example we use this `experiment.yaml`:
```yaml
task:
  name:
    - mnist
    - classification_small
optimizer:
  - name: adamw_baseline
    beta2: 0.98
  - name: sgd_baseline
    momentum: 0.5
engine:
  seed: [42, 47]
```
This will produce 2x2x2=8 runs in total.
Each undefined parameter will be set using either `engine/default.yaml`, `optimizers/<optimizer>/default.yaml` or `tasks/<task>/default.yaml`.

Before you run the experiment make sure the datasets are prepared:
```bash
python dataset_setup.py experiment.yaml
```

Then you run the experiment:
```bash
python experiment_runner.py experiment.yaml
```
This runs all tasks with all optimizers and hyperparameter specified inside `experiment.yaml` using grid-search.
You can either supply one value or a list of values for each entry. Grid-search combines each possible combination.  
For example: you specified 3 task, 2 optimizer, 2 different learning rates and 4 seeds then you need a total 3 x 2 x 2 x 4 = 48 runs

You can additionally set values trough the command line (this overrides existing values). For example you can set the `data_dir` where datasets are stored using either:
```bash
python "<script>.py" experiment.yaml "engine.data_dir=<path>"
```
or you can specify it inside the `experiment.yaml`:
```yaml
engine:
  data_dir: <path>
```
