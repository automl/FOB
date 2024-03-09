# Plotting

In the sample-data directory you can see which files are expected for creating plots.

Usually it is enough to call the plotting script with a path those results

```python plot.py -d sample-data/test-submission/test-workload/```

The script will (often succesfully) try to infer as many options as possible automatically from the given information. If it has to extrapolate information it will return a warning. 
for additional functionality see below and refer to
```python plot.py --help``` to see how to give the information if it cannot be inferred correctly.

# Usage Examples

the following will create a plot for all the trials of *sgd_baseline* on *cora* (all the trials in that directory), by default output will be placed in this directory

```
python plot.py -d $(ws_find bigdata)/experiments/sgd_baseline/cora 
```

-------------------------

The ```--depth``` flag can be used when your trials are not in the same directory. If they are organized as they are in the ```sorted_folder_example``` the search for trials can be set to a depth of 2 
```
python plot.py -d sample-data/example_submission/sorted_folder_example/ --depth 2
```

-------------------------

The ```--csv``` flag will create a file with the raw data

```
python plot.py -d sample-data/example_submission/ --csv
```

-------------------------

The ```--format``` flag changes the *precision* of the displayed result. default is ```2.3``` where the 2 changes the result to % and the number after the *dot* are the decimal places to display.
e.g. ```0.5``` will show the same digits but as accuracy instead of %

```
python plot.py -d sample-data/example_submission/ --format 0.5
```

-------------------------

The ```--std``` flag will add the standard deviation to the plot.

```
python plot.py -d sample-data/example_submission/ --std
```
