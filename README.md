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

## Workloads

We try to cover a large range of deep learning tasks in this benchmark.
 
| Dataset | Model | Task | Target Metric | Baseline Score | Baseline Runtime | Hardware |
| ------- | ----- | ---- | ------------- | -------------- | ---------------- | -------- |
| [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) | [Resnet18](https://arxiv.org/pdf/1512.03385.pdf) | Image classification | Validation Accuracy | 0.74 | 10 min | 1 gpu |
| [COCO](https://cocodataset.org) | [Faster R-CNN](https://arxiv.org/abs/1506.01497) with [MobileNet v3](https://arxiv.org/abs/1905.02244) backbone | Object detection | Average Precision | ? | ~4h | 4 gpus |
| [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/index.html) | ViT | Semantic Segmentation | Average Precision | ? | ? | ? |
| WMT | transformer | machine translation | ? | ? | ? | ? |
| openwebtext | transformer | unsupervised pretraining | ? | ? | ? | ? |
| librispeech | conformer | speech recognition | ? | ? | ? | ? |
| OGBG | graph NN | graph property prediction | ? | ? | ? | ? |
| FFHQ | ? | image diffusion | ? | ? | ? | ? |
| MNIST | MLP | Image classification | Validation Accuracy | 0.95 | 1 min | 1 gpu |
| California Housing | FT Transformer | tabular regression | Test MSE | 0.11 | 2 min | 1 gpu |
| ? | ? | Finetuning | ? | ? | ? | ? |

## Submissions

A submission contains the deep learning training algorithm to benchmark. Each submission has its own subfolder in the `submissions` folder. It can then be selected by the name of that folder with tha `-w` flag of the `submission_runner.py` script.  
Inside this folder you can put all the code necessary for your submission. It is required to put a `submission.py` file which contains two things. First, a class to encapsulate the submission. It must inherit from the `Submission` base class in `submissions/submissions.py` and provide a method `configure_optimizers` which returns the optimizers and LR schedulers to be used. The methods return type must adhere to the [API](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers) of a `LightningModule`. Second, a function `get_submission`, which returns an instance of your submission class.  
Have a look at the template and baseline submissions for examples of how to structure a submission.
