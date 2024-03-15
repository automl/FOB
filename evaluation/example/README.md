# Data creation

Lets create some data that we can plot; from the root directory call:

```
python experiment_runner.py evaluation/example/example_experiment_adamw.yaml
```

This will download and run the mnist task into the ```evaluation/example/``` directory (path can be changed in ```evaluation/example/example_experiment_adamw.yaml``` if you have already set up your benchmark).

Estimated disk usage: ~93MB

- data: ~64M
- experiments: ~29M

The task will be run on 2x2 hyperparameter on 3 different seeds for a total of 12 times.

Estimated time to finish: ~17min (TODO: might be 3x longer now)

# Plotting the experiment

To plot the performance heatmap, you can use the following from the root directory:

```
python evaluate_experiment.py evaluation/example/example_plot_instructions.yaml
```

Do not forget to adapt the path to the data that should be plotted if you customized it in step 1

Have a look inside the ```example_plot_instructions.yaml```. Not mentioned parameter are left on the defaults given in (```evaluation/default.yaml```).

This should create the following files:

- ```mnist_example-0.csv```
- ```mnist_example-heatmap.pdf```
- ```mnist_example-heatmap.png```

![your plot is not finished yet](mnist_example-heatmap.png)

# Comparing two experiments

You can also plot two experiments into the same file for easier comparison.

In this example we are going to compare SGD to AdamW; you can run the following command to get the additional data:

```
python experiment_runner.py evaluation/example/example_experiment_sgd.yaml 
```

The 2nd instrucion file has the path already set up, simply call:

```
python evaluate_experiment.py evaluation/example/example_plot_instructions2.yaml
```

have a look into the ```evaluation/example/example_plot_instructions2.yaml``` to see useful setting for this scenario


![your plot is not finished yet](<adamw vs sgd-heatmap.png>)