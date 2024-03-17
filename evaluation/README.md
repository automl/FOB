# Evaluation

During training you can monitor your experiments with [Tensorboard](https://www.tensorflow.org/tensorboard).  
We also try to provide some usefull functionality to quickly evaluate and compare the results of your experiments.

The usage is consistent with the benchmark; just as the ```experiment_runner.py``` the [```evaluate_experiment.py```](../evaluate_experiment.py) expects a ```.yaml``` file with a path to your experiment results and will overwrite any default option given in the [```default.yaml```](default.yaml)

You can use the plotting pipeline with your customized setting (as explained below) or alternatively you can use the script to export your data to a ```.csv``` and process the data to your own needs.

## csv

```bash
python evaluate_experiment.py csv.yaml
```

A simple ```csv.yaml``` just as in this example will scan the output in the experiment directory and summarize the data in a `.csv`

```yaml
data_dirs: path-to-your/experiments/task/optimizer  # the parent folder of the trials
output_dir: path-to-your/evaluation/example  # output filename is output_dir / experiment_name
experiment_name: csv_example  # 
output_types: [csv]  # no plotting, just the data
```

Alternatively you can call the `experiment_runner.py` with the already existing [`default.yaml` of the evaluation directory](default.yaml) and give your custom parameter directly in your terminal or bash script.

```bash
python evaluate_experiment.py evaluation/default.yaml "data_dirs=path-to-your/experiments/task/optimizer" "output_dir=path-to-your/evaluation/example" "experiment_name=csv_example" "output_types=[csv]"
```

## Heatmap plot

By adding ```pdf```, ```png``` to the ```output_types``` key or by replacing it to ```output_types=[pdf, png]``` an (additional) heatmap plot will be saved in the output directory.

There are numerous additional keys than can be given to finetune your plot.  
Have a look into the base settings to see the full list and their explanation.
[```evaluation/default.yaml```](default.yaml)

Just some examples:

- configure the functional details in the `plot` dictionary
    - which metric should appear on which axis
    - the metric that is plotted as score
- fontsize, colorpalette etc are in `plotstlye`
- a mapping for custom capitalization or math symbols can be given in `names`


## Usage Examples

There is a step by step example to plot mnist data in the [example subfolder](example/README.md)

There is a working example how to create multiple plots at the same time with a bash script in the [baselines subdirectory](./baselines)

