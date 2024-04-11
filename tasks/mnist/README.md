# task

Image Classification of handwritten digits on the famous MNIST dataset.

## dataset

Database of handwritten digits - size normalized and centered in 28x28 fixed size greyscale images
Using the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset as provided by [torchvision](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html).
60,000 examples (and additionally 10,000 for testing)

## model

A simple 2-layer NN, the size of the hidden dimension and the activation function are customizable.

## performance

![](../../evaluation/example/mnist_example-heatmap.png)

### performance comparison

[LeCun et al. 1998](https://ieeexplore.ieee.org/document/726791) reported test error rate for various neural networks (an overview can be found [on the MNIST website](http://yann.lecun.com/exdb/mnist/)). Comparable networks have an error rate of roughly 1% to 5%