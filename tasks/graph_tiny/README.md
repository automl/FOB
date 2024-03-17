# task

node classification on the cora dataset.

![](https://production-media.paperswithcode.com/datasets/Cora-0000000700-ce1c5ec7_LD7pZnT.jpg)

image source: https://arxiv.org/abs/1611.08402

## dataset

The cora dataset consists of a single network of 2708 publications classified into one of seven classes and consist of 5429 links.

https://paperswithcode.com/sota/node-classification-on-cora

https://link.springer.com/article/10.1023/A:1009953814988


## model

Here we use a GCN (Graph Convolutional Network)


## performance

The model achieves a performance of 80%

### performance comparison

Performance using a GCN was reported in 
https://paperswithcode.com/paper/semi-supervised-classification-with-graph

The authors report a classificationa accuracy of 81.5%

Their GCN has hyperparameter:
 - hidden_channel: 16
 - dropout: 0.5
and state that:
> or the citation network datasets, we optimize hyperparameters on Cora only and use the same set
> of parameters for Citeseer and Pubmed. We train all models for a maximum of 200 epochs (training
> iterations) using Adam (Kingma & Ba, 2015) with a learning rate of 0.01 and early stopping with a
> window size of 10, i.e. we stop training if the validation loss does not decrease for 10 consecutive
> epochs. We initialize weights using the initialization described in Glorot & Bengio (2010) and
> accordingly (row-)normalize input feature vectors.
