# task

Molecular Property Prediction (Graph Property Prediction) on the ogbg-molhiv dataset. The task is to predict the target molecular properties as accurately as possible: wheter a molecule inhibits HIV replication or not.

## dataset

[you can use the visualize.ipynb to look at the data](visualize.ipynb)

![molecule image from the notebook](random_molecule.png)

[ogb-molhiv](https://ogb.stanford.edu/docs/graphprop/#ogbg-mol) is adapted from [MoleculeNet](https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a)

nodes are atoms, and edges are chemical bonds, exact feature desciptions can be found in the [ogb code](https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py)

## model

The model for this task is a *Graph Isomorphism Network* (GIN) from [How powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826v3) which is a Graph Neural Network (GNN).

The model we use is the same as the one used as baseline on the official [OGB Leaderboard](https://ogb.stanford.edu/docs/leader_graphprop/) by the authors of the dataset.

## performance

The **metric** used is `ROC_AUC`.

### performance comparison

In [Semi-Supervised Classification with Graph Convolutional Networks]((https://arxiv.org/abs/1609.02907)), and on their [Leaderboard](https://ogb.stanford.edu/docs/leader_graphprop/) the OGB team reports a
- `Test ROC-AUC` of 0.7558 ± 0.0140 and a
- `Validation ROC-AUC` of 0.8232 ± 0.0090
