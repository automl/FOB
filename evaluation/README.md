# Evaluation

During training you can monitor your experiments with [Tensorboard](https://www.tensorflow.org/tensorboard).  
We also try to provide some usefull functionality to quickly evaluate and compare the results of your experiments.

One can use the ```evaluate_experiment.py``` or the ```experiment_runner.py``` to get a quick first impression of a finished experiment run.  

In the following you can find 4 example use cases for experiments and how to visualize the results as heatmaps.

## CSV

You can use the plotting pipeline with your customized setting (as shown in the usage examples).
alternatively you can use the script to export your data to a .csv and process the data to your own needs.

In this scenario, set ```evaluation.output_types: [csv]  # no plotting, just the data``` in your experiment yaml.

## Usage Examples

1. testing an optimizer on a task
2. comparing two optimizer on the same task
3. comparing the influence of single hyperparameter
4. comparing multiple optimizer on different tasks

### Data creation

Lets create some data that we can plot; from the root directory call:

#### Data Download

first we make sure the data is already downloaded beforehand:

```python dataset_setup.py evaluation/example/4_mnist-and-tabular_adamw-vs-sgd.yaml```

This will download the mnist data (required for 1-4) and tabular (required for 4) into the a ```evaluation/example/data``` directory - path can be changed in the correspong yaml you want to use (e.g.```evaluation/example/1_mnist-adamw.yaml``` if you have already set up your benchmark).

Estimated disk usage for the data: ~65M

#### Training

The task will be run on 2x2 hyperparameter on 2 different seeds per optimizer for a total of 8 times.

```python experiment_runner.py evaluation/example/1_mnist-adamw.yaml```

Here we do not really care for the performance; alternatively remove the ```task.max_epochs: 1``` argument in the yaml. This will perform a full training on the default epochs.

After training finished you should find 8 trial directories in ```evaluation/example/experiments/mnist/adamw_baseline```

All parameters that differ from the default value are noted in the directory name.

### Plotting the experiment

If not disabled in the experiment yaml, plotting the performance heatmap will be done together with training and testing.  

Alternatively you can call the ```evaluate_experiment.py``` from the root directory:

```python evaluate_experiment.py evaluation/example/1_mnist-adamw.yaml```


Have a look at the values given under the ```evaluation``` key. Not mentioned parameter are either set per task (e.g. in ```bob/tasks/mnist/default.yaml```) or left on the defaults given in (```evaluation/default.yaml```).

You can find the output in the newly created directory ```evaluation/example/plots```


You can use the csv to run the data through your own custom plotting workflow.

### Example use cases

Make sure to scroll through the ```evaluation``` key in the default.yaml if you are looking to adapt your plots; some of these keys are explained below the plots to give the reader a first impression.

#### 1

This example is a good starting point; it shows the performance of a single default optimizer on one of the tasks.

```python experiment_runner.py evaluation/example/1_mnist-adamw.yaml```

![your plot is not finished yet](plots/1_mnist-adamw.png)

Only use the final model performance and only create the plot as png.

- ```evaluation.checkpoints: [last]```  # you could use [last, best] to additionaly plot the model with the best validation
- ```output_types: [png]```  # you could use [pdf, png] to also create a pdf


#### 2

You can compare two different optimizer.

```python experiment_runner.py evaluation/example/2_adamw-vs-sgd.yaml```

![your plot is not finished yet](plots/2_adamw-vs-sgd.png)

Helpful settings:

- ```evaluation.plot.x_axis: [optimizer.weight_decay, optimizer.kappa_init_param]```  # the values given here are used as the value for the axis. The order in the list is used from left to right for the plot columns


#### 3

The square of a heatmap show the *mean* and *std* over multiple

```python experiment_runner.py evaluation/example/3_adamw-vs-sgd_seeds.yaml```

![your plot is not finished yet](plots/3_adamw-vs-sgd_seeds.png)

Helpful settings:

- Control the std with
    - ```plot.std```  # toggle off with ```False```
    - ```plot.aggfunc: std```  # also try ```var```
- control the rows with
    - ```split_groups: ["engine.seed"]```
    - ```aggregate_groups: []```

Per default (```split_groups: True```) the plot will display the *mean* calculated over *any* parameter that is neither on the x-axis nor y-axis.  
Every non unique value for each value in this list will create a own row.
This list is useful if there are just a few values to *not* aggreagte over

We need to remove the seed from the ```aggregate_groups``` list (by giving an empty list instead). This list is useful if there are just a few values to aggreagte over.

#### 4

There are multiple tasks in the benchmark, this example shows how to get a quick overview over multiple at the same time.

```python experiment_runner.py evaluation/example/4_mnist-and-tabular_adamw-vs-sgd.yaml```

![your plot is not finished yet](plots/4_mnist-and-tabular_adamw-vs-sgd.png)

Helpful settings:

 - ```split_groups: ["task.name"]```

 Just as in the seed example we explicitely add want to group by a specific value.

