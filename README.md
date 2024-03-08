# Fast Optimization Benchmark

Fast and cheap Benchmark for HPO and Optimizer.

Master Project at ML chair Freiburg,
Simon, Tobi, Zachi

This benchmark aims to be fast while maintaining a wide selection of different tasks. It also tries to be independant of the hardware used, however it requires a minimum of 4 gpus. One run of all workloads on a single submission and hyperparameter configuration should not take more than a day. A benchmark run returns the following for each workload: time taken per optimization step compared to baseline, best model performance, final model performance. 

## Installation

This repo was tested with python 3.10, but any version >= 3.8 should work.  
Create conda environment:
```
conda create -n fob python 3.10 -y
```
Activate and install requirements
```
conda activate fob
pip install -r requirements.txt
```

## Getting Started

Before running a workload you should first download the required data as this can take a while. Example usage:
```
python dataset_setup.py -w mnist
```
You can specify the location with `-d /path/to/data`. There is also the `--all` flag if you want all available workloads.

The benchmark consists of different workloads. You can run a certain workload with the `submission_runner.py` script. Example usage:
```
python submission_runner.py -w mnist -s adamw_baseline
```
This runs the `adamw_baseline` submission on the `mnist` workload. You can specify other things like the data and output locations, just run `python submission_runner.py -h` for more information.

## Tasks

We try to cover a large range of deep learning tasks in this benchmark.  
Some of these are still under development.

### Ready to use

| Name | Dataset | Model | Task | Target Metric | Baseline Score | Baseline Runtime | Hardware |
| ------- | ---- | ----- | ---- | ------------- | -------------- | ---------------- | -------- |
| mnist | MNIST | MLP | Image classification | Validation Accuracy | 0.95 | 1 min | 1 gpu |
| classification_small | [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) | [Resnet18](https://arxiv.org/pdf/1512.03385.pdf) | Image classification | Top-1 Accuracy | 0.74 | 10 min | 1 gpu |
| classification_large | [Imagenet-64x64](https://patrykchrabaszcz.github.io/Imagenet32/) | [Wide ResNet](https://arxiv.org/pdf/1605.07146.pdf) | Image classification | Top-1 Accuracy | 0.69 | 4h | 4 gpu |
| segmentation | [MIT Scene Parse](http://sceneparsing.csail.mit.edu/) | [SegFormer](https://arxiv.org/abs/2105.15203) | Semantic Segmentation | Average Precision (IoU) | 0.26 | 4h | 1 gpu |
| graph | [ogbg-molhiv](https://ogb.stanford.edu/docs/graphprop/#ogbg-mol) | [Graph Isomorphism Network (GIN)](https://arxiv.org/pdf/1810.00826.pdf) | graph property prediction | ROC-AUC | 0.75 | ? | 1 gpu |
| tabular | [California Housing](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html) | [FT Transformer](https://arxiv.org/pdf/2106.11959.pdf) | tabular regression | Test MSE | 0.11 | 2 min | 1 gpu |



### Under Development

| Name | Dataset | Model | Task | Target Metric | Baseline Score | Baseline Runtime | Hardware |
| ------- | ----- | ----- | ---- | ------------- | -------------- | ---------------- | -------- |
| detection | [COCO](https://cocodataset.org) | [Faster R-CNN](https://arxiv.org/abs/1506.01497) with [MobileNet v3](https://arxiv.org/abs/1905.02244) backbone | Object detection | Average Precision (IoU) | ? | ~4h | 4 gpus |
| translation | WMT17(de-en) | transformer | machine translation | cross-entropy loss / BLEU score | ? | 4h | 4 gpus |
| graph_tiny | [cora](https://paperswithcode.com/sota/node-classification-on-cora) | [GCN](https://arxiv.org/abs/1609.02907) | Node Classification | test accuracy | TODO (in paper 81.5) | TODO (very fast, just some minutes) | 1 gpu |
| speech | librispeech | conformer | speech recognition | ? | ? | ? | ? |
| diffusion | FFHQ | ? | image diffusion | ? | ? | ? | ? |
| | ? | ? | Finetuning | ? | ? | ? | ? |


## Submissions

A submission contains the deep learning training algorithm to benchmark. Each submission has its own subfolder in the `submissions` folder. It can then be selected by the name of that folder with tha `-w` flag of the `submission_runner.py` script.  
Inside this folder you can put all the code necessary for your submission. It is required to put a `submission.py` file which contains two things. First, a class to encapsulate the submission. It must inherit from the `Submission` base class in `submissions/submissions.py` and provide a method `configure_optimizers` which returns the optimizers and LR schedulers to be used. The methods return type must adhere to the [API](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers) of a `LightningModule`. Second, a function `get_submission`, which returns an instance of your submission class.  
Have a look at the template and baseline submissions for examples of how to structure a submission.

## Usage Instructions

### How to run a workload on a suite of hyperparameters
Make sure you have the conda environment set up and activated. Then run the following commands.  
The dataset setup should be run beforehand. You can restrict the number of workers with `--workers` if you dont have a lot of cpus available (e.g. on the login node of a cluster).
```
python dataset_setup.py -w <WORKLOAD> -d <DATA_DIR>
```
Where `<DATA_DIR>` will be the folder where the data is stored.

If you want to cover a search space, you need to generate the hyperparameter files.
```
python suite_generator.py -s <SEARCH_SPACE> -o <HYPERPARAMETER_DIR>
```
Where `<SEARCH_SPACE>` is a json file (as in the baseline submissions) and `<HYPERPARAMETER_DIR>` is the folder where the hyperparameter files will be stored.

The command for actually running the workload is as follows (you can omit `srun` if you do not use SLURM).
```
srun python submission_runner.py -d <DATA_DIR> -o <OUTPUT_DIR> -w <WORKLOAD> -s <SUBMISSION> --hyperparameters <HYPERPARAMETER_DIR> --trials <TRIALS>
```
Where `<DATA_DIR>`, `<WORKLOAD>` and `<SUBMISSION>` and `<HYPERPARAMETER_DIR>` are the same as before, while
- `<OUTPUT_DIR>` is the directory where the results are stored
- `<TRIALS>` is the number of trials to run (needs to be equal to the number of hyperparameter files).
If you want to run an array job, with one job per hyperparameter, you need to adjust the `--start_trial` and `--start_hyperparameter` flags instead.
