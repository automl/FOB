# Data creation

Lets create some data that we can plot; from the root directory call:

## Data Download

(first make sure the data is already downloaded beforehand)
```
python dataset_setup.py evaluation/example/example_experiment_adamw.yaml
```
This will download the mnist task into the ```evaluation/example/``` directory (path can be changed in ```evaluation/example/example_experiment_adamw.yaml``` if you have already set up your benchmark).

Estimated disk usage for the data: ~64M

## Training

The task will be run on 2x2 hyperparameter on 2 different seeds for a total of 8 times.

```
python experiment_runner.py evaluation/example/example_experiment_adamw.yaml task.max_epochs=1
```

Here we do not really care for the performance; alternatively remove the ```task.max_epochs=1``` argument like in the following command. this will perform a full training on the default epochs 

```
python experiment_runner.py evaluation/example/example_experiment_adamw.yaml 
```

After training finished you should find 12 trial directories in ```evaluation/example/experiments/mnist/adamw_baseline```

# Plotting the experiment

To plot the performance heatmap, you can use the following from the root directory:

```
python evaluate_experiment.py evaluation/example/plot_instructions.yaml
```

Do not forget to adapt the path to the data that should be plotted if you customized it in step 1

Have a look inside the ```plot_instructions.yaml```. Not mentioned parameter are left on the defaults given in (```evaluation/default.yaml```).

This should create the following files:

- ```mnist_example-0.csv```
- ```mnist_example-heatmap.pdf```
- ```mnist_example-heatmap.png```

You can use the csv to run the data through your own custom plotting workflow.

![your plot is not finished yet](mnist_example-heatmap.png)

# Comparing two experiments

You can also plot two experiments into the same file for easier comparison.

In this example we are going to compare SGD to AdamW; you can run one of the following commands to get the additional data:

```
python experiment_runner.py evaluation/example/example_experiment_sgd.yaml task.max_epochs=1
```

The second instrucion file has the path already set up, to get a plot you can simply call:

```
python evaluate_experiment.py evaluation/example/plot_instructions2.yaml
```

have a look into the ```evaluation/example/plot_instructions2.yaml``` to see useful setting for this scenario

![your plot is not finished yet](<adamw vs sgd-heatmap.png>)
