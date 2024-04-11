# task

Image Classification on the ImageNet-64 Dataset-Variant.

## dataset

[Imagenet-64](https://paperswithcode.com/dataset/imagenet-64) is a down-sampled variant of [ImageNet](https://www.image-net.org/).
It comprises 1,281,167 training data and 50,000 test data with 1000 labels.

![](https://patrykchrabaszcz.github.io/assets/img/Imagenet32/64x64.png)
image source: https://patrykchrabaszcz.github.io/Imagenet32/

## model

We want to follow [A DOWNSAMPLED VARIANT OF IMAGENET AS AN ALTERNATIVE TO THE CIFAR DATASETS](https://arxiv.org/pdf/1707.08819v3.pdf) ([papers with code](https://paperswithcode.com/paper/a-downsampled-variant-of-imagenet-as-an))

They use Wide Residual Networks WRN-N-k by [Zagoruyko and Komodakis (2016)](https://arxiv.org/abs/1605.07146), the [original model and code is given in lua](https://github.com/szagoruyko/wide-residual-networks/blob/master/models/wide-resnet.lua)

which have been slightly adapted:
> To adapt WRNs for images with 64 × 64 pixels per image as used
in ImageNet64x64, we add an additional stack of residual blocks to reduce the spatial resolution of
the last feature map from 16 × 16 to 8 × 8 and thus double the number of features

This adaption was done using [lasagna to build Theano networks](https://lasagne.readthedocs.io/en/latest/)

To get our implementation we build the same model as in the [imagenet64 code](https://github.com/PatrykChrabaszcz/Imagenet32_Scripts) using an adaption of a [wrn implementation](https://github.com/osmr/imgclsmob/blob/c03fa67de3c9e454e9b6d35fe9cbb6b15c28fda7/pytorch/pytorchcv/models/wrn.py#L239)

### wrn
Using the [```wise_resnet50_2``` torchvision model](https://pytorch.org/vision/main/models/generated/torchvision.models.wide_resnet50_2.html) from
[Wide Residual Networks](https://arxiv.org/abs/1605.07146)

### davit

[DaViT](https://arxiv.org/abs/2204.03645)

## performance

### performance comparison

using a WRN (N=36, k=5): top 1 Error 32,34%  by the [imagenet64 authors](https://paperswithcode.com/sota/image-classification-on-imagenet-64)