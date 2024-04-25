# Task

The task is to predict the price of houses. 

## Dataset

We use the [california housing](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html) dataset. It has 8 numerical and no categorical features. The total size of the dataset is 20640. The test split used for evaluation is the same as in [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959v5).

## Model

We use the [FT-Transformer](https://arxiv.org/abs/2106.11959v5) model with its default parameter settings.

## Performance

We compare the Root Mean Squared Error (RMSE). Our model reaches a performance of `0.397 ± 0.006` RMSE. The search grid used to find the optimal hyperparameters can be found [here](../../baselines/tabular.yaml).

### Performance Comparison

We compare our model against the performance reported in [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959v5). There the authors report a performance of `0.459` RMSE (see Table 2 of the paper). Reproducing the exact hyperparameters the authors used is difficult as the authors used [Optuna](https://optuna.org/) to optimize the hyperparameters and did not state the optimal hyperparameters found. Using their default hyperparameter settings, we achieve an RMSE of `0.404±(0.004)`. This difference might be explained by the choice of preprocessing used. While the authors state that they use sklearns [QuantileTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html), the performance achieved in the paper is closer to what we acieve with the [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html). When preprocessing with the [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) on their default hyperparameters, we obtain a performance of `0.453±(0.015)`, which is much closer to the reported performance. To reproduce the values for this comparison, use the following config:
```yaml
task:
  name: tabular
  output_dir_name: tabular_reference
  train_transforms:
    - normalizer: quantile
      noise: 1.e-3
    - normalizer: standard
      noise: 0.0
optimizer:
  name: adamw_baseline
  learning_rate: 1.e-4
  weight_decay: 1.e-5
  eta_min_factor: 1.0
engine:
  seed: [1, 2, 3]
```
