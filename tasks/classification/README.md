# Task

Image Classification on the ImageNet-64 Dataset-Variant.

## Dataset

[Imagenet-64](https://paperswithcode.com/dataset/imagenet-64) is a down-sampled variant of [ImageNet](https://www.image-net.org/).
It comprises 1,281,167 training data and 50,000 test data with 1000 labels.

![](https://patrykchrabaszcz.github.io/assets/img/Imagenet32/64x64.png)
image source: https://patrykchrabaszcz.github.io/Imagenet32/

## Model

We want to follow [a downsampled variant of imagenet as an alternative to the cifar datasets](https://arxiv.org/pdf/1707.08819v3.pdf).

They use Wide Residual Networks WRN-N-k by [Zagoruyko and Komodakis (2016)](https://arxiv.org/abs/1605.07146), the [original model and code is given in lua](https://github.com/szagoruyko/wide-residual-networks/blob/master/models/wide-resnet.lua)

Since the implementation of the Imagenet-64 authors uses the [lasagne library](https://lasagne.readthedocs.io/en/latest/) it is not really compatible with our framework. Therefore we use the closest implementation from [torchvision](https://pytorch.org/vision/stable/index.html).

### Wide ResNet
The best performing Configuration in the paper was N=36, k=5. The closest available implementation from [torchvision](https://pytorch.org/vision/stable/index.html) was the [```wide_resnet_50_2``` torchvision model](https://pytorch.org/vision/main/models/generated/torchvision.models.wide_resnet50_2.html), with N=50, k=2.

### DaViT
We also tried to use a vision transformer model, namely [DaViT](https://arxiv.org/abs/2204.03645). However, this did not result in a satisfying performance.

## Performance
We compare the top 1 accuracy. Our model achieved a top 1 accuracy of around `69`% (no exact result, experiments are still running). The search grid used to find the (currently) best hyperparameters can be found [here](../../baselines/classification.yaml).

### Performance comparison
The report a top 1 accuracy of `67.66`% using a WRN (N=36, k=5). We tried to replicate their performance by using the same hyperparameters, which can be found [here](reference.yaml)
