# Task

Molecular Property Prediction (Graph Property Prediction) on the ogbg-molhiv dataset. The task is to predict the target molecular properties as accurately as possible: wheter a molecule inhibits HIV replication or not.

## Dataset

[ogb-molhiv](https://ogb.stanford.edu/docs/graphprop/#ogbg-mol) is adapted from [MoleculeNet](https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a) and was introduced in [Open Graph Benchmark: Datasets for Machine Learning on Graphs (NeurIPS 2020)](https://paperswithcode.com/paper/open-graph-benchmark-datasets-for-machine)
and is part of the [Open Graph Benchmark (OGB)](https://ogb.stanford.edu/)

The nodes are atoms, and the edges of the graph are chemical bonds, exact feature desciptions can be found in the [ogb code](https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py).

For more information (e.g. molecule name, chemical formula, figures) on the molecules in the dataset we provide a [.ipynb notebook](visualize.ipynb).

![E404: follow the notebook for a plot of a molecule!](random_molecule.png)

## Model

The model for this task is a *Graph Isomorphism Network* (GIN) from [How powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826v3) which is a Graph Neural Network (GNN).

The model we use is the same as the one used as baseline on the official [OGB Leaderboard](https://ogb.stanford.edu/docs/leader_graphprop/) by the authors of the dataset.

## Performance

The performance is measured in `ROC_AUC` (receiver operating characteristic - area under curve): higher is better and the range of values is [0, 1].

Our model reaches a performance of `0.774 ± 0.0107`. The search grid used to find the optimal hyperparameters can be found [here](../../baselines/graph.yaml).

### Performance Comparison

In [Semi-Supervised Classification with Graph Convolutional Networks]((https://arxiv.org/abs/1609.02907)), and on their [Leaderboard](https://ogb.stanford.edu/docs/leader_graphprop/) the OGB team reports a `Test ROC-AUC` of `0.7558 ± 0.0140`.

## Additional Information

During development training was numerically unstable for learning rates > 1.0e-3 when using 16bit Automatic Mixed Precision (AMP).
Make sure to try (a machine with) bf16 precision if you run into similar issues.
