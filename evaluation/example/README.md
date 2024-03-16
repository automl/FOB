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

alternatively add ```task.max_steps=108``` like in the following command. this will reduce the train time (which is fine since we do not really care for performace here) TODO: change this to ```task.max_epochs=1``` once steps are calcualted automatically

```
python experiment_runner.py evaluation/example/example_experiment_adamw.yaml task.max_steps=108
```


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
python experiment_runner.py evaluation/example/example_experiment_sgd.yaml 
```

TODO: as above, change to ```task.max_epoch=1```

```
python experiment_runner.py evaluation/example/example_experiment_sgd.yaml task.max_steps=108
```

The second instrucion file has the path already set up, to get a plot you can simply call:

```
python evaluate_experiment.py evaluation/example/plot_instructions2.yaml
```

have a look into the ```evaluation/example/plot_instructions2.yaml``` to see useful setting for this scenario

![your plot is not finished yet](<adamw vs sgd-heatmap.png>)
