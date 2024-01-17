import json
import os, sys
from pathlib import Path
import argparse
import re
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
import multiprocessing as mp
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def draw_heatmap(df, ax, values, index, columns, std=False):
    pivot_table = pd.pivot_table(df, values=values, index=index, columns=columns, aggfunc='mean')
    pivot_table = pivot_table * 100
    pivot_table = pivot_table.round(0)
    if not std:
        sns.heatmap(pivot_table, annot=True, fmt=".0f", ax=ax, annot_kws={'fontsize': 8}, cbar_ax=cbar_ax, vmin=60, vmax=80)
    else:
        pivot_table_std = pd.pivot_table(df, values=values, index=index, columns=columns, aggfunc='std')
        pivot_table_std = pivot_table_std * 100
        annot_matrix = pivot_table.copy()
        for i in pivot_table.index:
            for j in pivot_table.columns:
                mean = pivot_table.loc[i, j]
                std = pivot_table_std.loc[i, j]
                annot_matrix.loc[i, j] = f"{mean:.1f}\n±{std:.2f}"
        sns.heatmap(pivot_table, annot=annot_matrix, fmt="", annot_kws={'fontsize': 5}, ax=ax, cbar_ax=cbar_ax, vmin=60, vmax=80)
    return pivot_table


def get_available_trials(dirname: Path):
    """finds all trials in the subfolder"""
    subdirs = [x for x in dirname.iterdir() if x.is_dir()]
    format_string = "\n  "
    print(f"found the following directories:{format_string}{format_string.join(str(i) for i in subdirs)}.")

    def is_trial(path: Path):
        # here we could do additional checks to filter the subdirectories
        for x in path.iterdir():
            if x.name == "hyperparameters.json":
                return True
        return False
    
    subdirs = list(filter(is_trial , subdirs[::-1]))
    print(f"we assume the following to be trials:{format_string}{format_string.join(str(i) for i in subdirs)}.")
    return subdirs


def dataframe_from_trials(trial_dir_paths: List[Path]):
    """takes result from get_available_trials and packs them in a dataframe"""
    dfs = [] # an empty list to store the data frames
    for path in trial_dir_paths:
        hyperparameters_file = path / "hyperparameters.json"
        result_best_model_file = path / "results_best_model.json"
        seed_file = path / "runtime_args.json"

        with open(seed_file, 'r') as f:
            seed = json.load(f)["seed"]
        # print(seed)

        with open(result_best_model_file) as f:
            accuracy = json.load(f)[0]["test_acc"]
        # print(accuracy)

        with open(hyperparameters_file) as f:
            data = pd.json_normalize(json.loads(f.read()))
            data.at[0, "test_acc"] = accuracy  # will trim to 4 digits after comma
            data.at[0, "seed"] = seed  # saved as float
            # print(data)
            dfs.append(data) # append the data frame to the list
    df = pd.concat(dfs, sort=False)
    # print(df)
    return df


def create_matrix_plot(dataframe):
    pivot_table = pd.pivot_table(dataframe, index="learning_rate", columns="weight_decay", values= "test_acc", aggfunc='mean')
    pivot_table = (pivot_table * 100).round(0)
    print(pivot_table)
    # left bottom width height
    # cbar_ax = fig.add_axes([0.92, 0.235, 0.02, 0.6])
    cbar_ax = None
    ax = None
    return sns.heatmap(pivot_table, annot=True, fmt=".0f", ax=ax, annot_kws={'fontsize': 8}, cbar_ax=cbar_ax, vmin=60, vmax=80)


def create_figure(workload_paths: List[Path]):
    """Takes a list of workloads Paths (submission + workload)
    and plots them together in one figure side by side"""
    num_subfigures: int = len(workload_paths)

    # create list of subfigures
    # for each create a matrix plot
    for i in range(num_subfigures):
        available_trials = get_available_trials(workload_paths[i])
        dataframe = dataframe_from_trials(available_trials)
        current_plot = create_matrix_plot(dataframe)


if __name__ == "__main__":
    print("Plotting script was started.")
    # todo, take these as args
    data_dir = Path("/home/haeringz/bob/evaluation/sample-data")
    submission_name = "test-submission"
    workload_name = "test-workload"
    output_file = "/home/haeringz/bob/evaluation/test-plot.pdf"

    workloads: List[Path] = [data_dir / submission_name / workload_name]
    
    create_figure(workloads)
    plt.savefig(output_file)
