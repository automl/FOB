# folder structure of results

outputdir
    - <submission> (e.g. adamw)
        - <workloads> (e.g. mnist)
            - <trial_nr>

in the trial dir one can find a file for results and hyperparameter;  
aggregate over the seeds and hyperparameter found to create the matrix plot


# example image

It should look somethine like this
![Alt text](image.png)

python path_to_workload_of_submission_1 path_to_workload_of_submission_2

# Usage

```
python plot.py -d $(ws_find bigdata)/experiments/sgd_baseline/cora 
```

this will create a plot for all the trials of *sgd_baseline* on *cora* (all the trials in that directory), by default output will be placed in this directory
