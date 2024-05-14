# Task

Node Classification on the Cora Dataset.

![](https://production-media.paperswithcode.com/datasets/Cora-0000000700-ce1c5ec7_LD7pZnT.jpg)

image source: https://arxiv.org/abs/1611.08402

## Dataset

The cora dataset consists of a single network of 2708 publications classified into one of seven classes and consist of 5429 links.

It was originally prepared by [McCallum et al. 2000](https://link.springer.com/article/10.1023/A:1009953814988).

Here we use the planetoid version from [Revisiting Semi-Supervised Learning with Graph Embeddings](https://arxiv.org/abs/1603.08861)

[Paperswithcode](https://paperswithcode.com/sota/node-classification-on-cora)

## Model

Here we use a GCN (Graph Convolutional Network), which was first introduced by [Semi-Supervised Classification with Graph Convolutional Networks, Kipf & Welling 2017](https://arxiv.org/abs/1609.02907)

![](https://tkipf.github.io/graph-convolutional-networks/images/gcn_web.png)

(image source: https://tkipf.github.io/graph-convolutional-networks/)

## Performance

We compare the Accuracy. The search grid used to find the optimal hyperparameters can be found [here](../../baselines/graph_tiny.yaml).
Our model achieves a performance of `81.9 ± 0.6%` (using the best checkpoint).

### Performance Comparison

Performance using a GCN was reported in [Semi-Supervised Classification with Graph Convolutional Networks, Kipf & Welling 2017](https://arxiv.org/abs/1609.02907), where the authors report a classificationa accuracy of `81.5%`

Their GCN configuration was found by training models:
- 200 epochs (training iterations)
- early stopping: window size 10 (stop training if val loss does not decrease for 10 consecutive epochs)
- optimizer: adam
- weights are initialized as described in [Glorot & Bengio (2010)](https://proceedings.mlr.press/v9/glorot10a.html) and input feature vectors are (row-)normalize accordingly.

Their model:

- num layer: 2
- hidden_channel: 16
- dropout: 0.5
- learning rate 0.01
- L2 regularization: 5 · 10−4  (first GCN layer)
