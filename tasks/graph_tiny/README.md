# model
GCN = Graph Convolutional Networks 
https://paperswithcode.com/paper/semi-supervised-classification-with-graph

The authors report a classificationa accuracy of 81.5

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

we use a larger network

# dataset

cora 

https://paperswithcode.com/sota/node-classification-on-cora

https://link.springer.com/article/10.1023/A:1009953814988

# task

single graph, task is to classify the nodes

![](https://production-media.paperswithcode.com/datasets/Cora-0000000700-ce1c5ec7_LD7pZnT.jpg)

image source: https://arxiv.org/abs/1611.08402