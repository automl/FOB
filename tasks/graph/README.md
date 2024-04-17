# task

Molecular Property Prediction (Graph Property Prediction) on the ogbg-molhiv dataset. The task is to predict the target molecular properties as accurately as possible: wheter a molecule inhibits HIV replication or not.

## dataset

![you can use the visualize.ipynb to look at the data](random_molecule.png)

[ogb-molhiv](https://ogb.stanford.edu/docs/graphprop/#ogbg-mol) is adapted from [MoleculeNet](https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a)

nodes are atoms, and edges are chemical bonds, exact feature desciptions can be found in the [ogb code](https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py)

## model

We use a Graph Convolutional Network (GNN)

## performance

### performance comparison
