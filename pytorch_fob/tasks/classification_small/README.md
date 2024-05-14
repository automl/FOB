# Task

Image Classification on the Cifar-100 Dataset.

## Dataset

The [Cifar-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) consists of 60000 32x32 color images in 100 categories.

## Model

We use the [Resnet18](https://arxiv.org/pdf/1512.03385.pdf) model [implementation from torchvision](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).

## Performance

We compare the top-1 accuracy. Our model reaches a top-1 accuracy of `77.9
± 0.2`%. The search grid used to find the best hyperparameters can be found [here](../../baselines/classification_small.yaml).

### Performance comparison
