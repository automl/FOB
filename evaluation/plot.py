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


#### LOAD BASELINE AND BASE EXPT

dataframe_name = "dataframe_F.pd"
cpr_kf_dir = "/home/joerg/workspace/experiments/adamCPR_cifar100/cifar100_FINAL_all_adapt_4"
df = load_dataframe(dataframe_name, cpr_kf_dir)
print(f"Load {dataframe_name} Done")

# left bottom width height
cbar_ax = fig.add_axes([0.92, 0.235, 0.02, 0.6])


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
    ...


def dataframe_from_trials():
    """takes result from get_available_trials and packs them in a dataframe"""
    ...


def create_matrix_plot(dataframe):
    ...


def create_figure(workload_paths: List[Path]):
    """Takes a list of workloads Paths (submission + workload)
    and plots them together in one figure side by side"""
    num_subfigures: int = len(workload_paths)

    # create list of subfigures
    # for each create a matrix plot
    for i in range(num_subfigures):
        available_trials = get_available_trials()
        dataframe = dataframe_from_trials(available_trials)
        current_plot = create_matrix_plot(dataframe)


if __name__ == "__main__":
    ...
    # todo, take these as args
    data_dir = "./sample-data"
    submission_name = "test-submission"
    workload_name = "test-workload"

    workloads: List[Path] = list(data_dir / submission_name / workload_name)

    create_figure(workloads)
