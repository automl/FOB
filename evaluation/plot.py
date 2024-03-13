import json
import yaml
from pathlib import Path
import argparse
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# TODO: these should probably all be made into parser args, or dynamically extracted
SEED = "seed"
# default file names
RESULT_BEST_FILENAME = "results_best_model.json"
RESULT_LAST_FILENAME = "results_final_model.json"
CONFIG_FILENAME = "config.yaml"
DEFAULT_FILE_ENDING = "png"
OPTIM_TO_TITLE = {
    "adamw_baseline": "AdamW",
    "sgd_baseline": "SGD"
}


def get_available_trials(dirname: Path, depth: int = 1):
    """finds the path for all trials in the *dirname* directory"""
    # RECURSIVELY FIND ALL DIRS IN DIRNAME (up to depth)
    subdirs: list[Path] = [dirname]
    all_results_must_be_same_depth = True
    for _ in range(depth):
        if all_results_must_be_same_depth:
            new_subdirs: list[Path] = []
            for subdir in subdirs:
                new_subdirs += [x for x in subdir.iterdir() if x.is_dir()]
            subdirs = new_subdirs
        else:
            for subdir in subdirs:
                subdirs += [x for x in subdir.iterdir() if x.is_dir()]

    if args.verbose:
        format_string = "\n  "
        print(f"found the following directories:{format_string}{format_string.join(str(i) for i in subdirs)}.")

    def is_trial(path: Path):
        # here we could do additional checks to filter the subdirectories
        # currently we only check if there is a config file
        for x in path.iterdir():
            if x.name == CONFIG_FILENAME:
                return True
        return False

    subdirs = list(filter(is_trial, subdirs[::-1]))
    if args.verbose:
        print(f"we assume the following to be trials:{format_string}{format_string.join(str(i) for i in subdirs)}.")
    return subdirs


def dataframe_from_trials(trial_dir_paths: List[Path]):
    """takes result from get_available_trials and packs them in a dataframe,
    does not filter duplicate hyperparameter settings."""
    dfs: List[pd.DataFrame] = []
    stats: list[dict] = []

    for path in trial_dir_paths:
        stat = {}
        hp_dict = {}

        config_file = path / CONFIG_FILENAME
        result_file = path / RESULT_BEST_FILENAME
        if args.last_instead_of_best:
            result_file = path / RESULT_LAST_FILENAME
        all_files_exist = all([
            config_file.is_file(),
            result_file.is_file()
        ])
        if not all_files_exist:
            print(f"WARNING: one or more files are missing in {path}. Skipping this hyperparameter setting.")
            continue

        with open(config_file, "r", encoding="utf8") as f:
            yaml_content = yaml.safe_load(f)
            if args.verbose:
                print(f"{yaml_content=}\n")
            stat["seed"] = yaml_content["engine"]["seed"]
            stat["task_name"] = yaml_content["task"]["name"]
            stat["optimizer_name"] = yaml_content["optimizer"]["name"]
            stat["target_metric_mode"] = yaml_content["task"]["target_metric_mode"]
            stat["target_metric"] = yaml_content["task"]["target_metric"]
            TARGET_METRIC_MODE = stat["target_metric_mode"]  # max or min
            hp_dict = yaml_content["optimizer"]
            
        with open(result_file) as f:
            content = json.load(f)
            if args.metric in content[0]:
                stat["score"] = content[0][args.metric]
            else:
                stat["metric"] = stat["target_metric"].replace("val_", "test_")
                print(f"WARNING: given metric '{args.metric}' does not exist, please check for typos!... " +
                      f"Using '{stat['metric']}' because '{stat['target_metric']}' was the target metric.")
                stat["score"] = content[0][stat["metric"]]

        data = pd.json_normalize(hp_dict)
        data.at[0, args.metric] = stat["score"]  # will trim to 4 digits after comma
        data.at[0, SEED] = stat["seed"]  # saved as float
        dfs.append(data)  # append the data frame to the list

        if args.verbose:
            print(f"{stat=}")
        stats.append(stat)

    df = pd.concat(dfs, sort=False)

    return df, stats


def create_matrix_plot(dataframe, ax=None, lower_is_better: bool = False, stat: dict={}):
    # create pivot table and format the score result
    pivot_table = pd.pivot_table(dataframe,
                                 columns=args.x_axis, index=args.y_axis, values=args.metric,
                                 aggfunc='mean')
    value_exp_factor, decimal_points = args.format.split(".")
    value_exp_factor = int(value_exp_factor)
    decimal_points = int(decimal_points)
    pivot_table = (pivot_table * (10 ** value_exp_factor)).round(decimal_points)
    if args.verbose:
        print(pivot_table)

    # setting the COLORMAP and the RANGE of the values for the colors (legend bar)
    vmin = args.limits and min(args.limits)  # lower limit (or None if not given)
    vmax = args.limits and max(args.limits)  # upper limit (or None if not given)
    colormap_name = "rocket"
    if lower_is_better:
        colormap_name += "_r"  # this will "inver" / "flip" the colorbar
    colormap = sns.color_palette(colormap_name, as_cmap=True)
    metric_legend = stat["metric"] if "metric" in stat.keys() else args.metric

    # FINETUNE POSITION
    # left bottom width height
    # cbar_ax = fig.add_axes([0.92, 0.235, 0.02, 0.6])
    cbar_ax = None

    if not args.std:
        return sns.heatmap(pivot_table, ax=ax, cbar_ax=cbar_ax,
                           annot=True, fmt=f".{decimal_points}f", annot_kws={'fontsize': 8},
                           vmin=vmin, vmax=vmax, cmap=colormap, cbar_kws={'label': f"{metric_legend}"})
    else:
        # PRECISION TO DISPLAY
        std_decimal_points = decimal_points  # good default would be 2
        mean_decimal_points = decimal_points  # good default would be 1

        # BUILD STD TABLE
        pivot_table_std = pd.pivot_table(dataframe,
                                         columns=args.x_axis, index=args.y_axis, values=args.metric,
                                         aggfunc='std')
        pivot_table_std = (pivot_table_std * (10 ** value_exp_factor)).round(decimal_points)
        annot_matrix = pivot_table.copy().astype("string")  # TODO check if this explicit cast is the best
        for i in pivot_table.index:
            for j in pivot_table.columns:
                mean = pivot_table.loc[i, j]
                std = pivot_table_std.loc[i, j]
                annot_matrix.loc[i, j] = f"{round(mean, mean_decimal_points)}\n±({round(std, std_decimal_points)})"

        fmt = ""  # cannot format like before, as we do not only have a number
        return sns.heatmap(pivot_table, ax=ax, cbar_ax=cbar_ax,
                           annot=annot_matrix, fmt=fmt, annot_kws={'fontsize': 5},
                           vmin=vmin, vmax=vmax, cmap=colormap, cbar_kws={'label': f"{metric_legend}"})


def create_figure(dataframe_list: list[pd.DataFrame], stats_list: list[dict]):
    """Takes a list of workloads Paths (submission + workload)
    and plots them together in one figure side by side"""
    num_subfigures: int = len(dataframe_list)

    # Create a 1x2 subplot layout
    n_rows = 1
    n_cols = num_subfigures

    # TODO, figsize is just hardcoded for (1, 2) grid and left to default for (1, 1) grid
    #       probably not worth the hazzle to create something dynamic (atleast not now)
    # margin = (num_subfigures - 1) * 0.3
    # figsize=(5*n_cols + margin, 2.5)
    figsize = None
    if num_subfigures == 2:
        figsize = None
        figsize = (12 * args.scale, 5.4 * args.scale)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    if num_subfigures == 1:
        axs = [axs]  # adapt for special case so we have unified types

    # Adjust left and right margins as needed
    # fig.subplots_adjust(left=0.1, right=0.9, top=0.97, hspace=0.38, bottom=0.05,wspace=0.3)
    for i in range(num_subfigures):
        dataframe, stats = dataframe_list[i], stats_list[i]
        some_stat_entry = stats[0]  # just get an arbitrary trial for the target metric mode and submission name
        opti_name = some_stat_entry['optimizer_name']
        s_target_metric_mode = some_stat_entry["target_metric_mode"]
        lower_is_better = s_target_metric_mode == "min"

        current_plot = create_matrix_plot(dataframe, ax=axs[i], lower_is_better=lower_is_better, stat=some_stat_entry)

        # Pretty name for label "learning_rate" => "Learning Rate"
        current_plot.set_xlabel(current_plot.get_xlabel().replace('_', ' ').title())
        current_plot.set_ylabel(current_plot.get_ylabel().replace('_', ' ').title())

        if i > 0:
            # remove y_label of all but first one
            axs[i].set_ylabel('', fontsize=8, labelpad=8)
        else:
            # TODO format parameter just as in submission name
            # axs[i].set_ylabel
            pass

        # title (heading) of the figure:
        title = OPTIM_TO_TITLE[opti_name] if opti_name in OPTIM_TO_TITLE.keys() else opti_name.replace('_', ' ').title()
        title += " on "
        title += some_stat_entry["task_name"]
        axs[i].set_title(title)

    fig.tight_layout()
    return fig, axs


def extract_dataframes(workload_paths: List[Path], depth: int = 1) -> tuple[list[pd.DataFrame], list[dict]]:
    df_list: list[pd.DataFrame] = []
    stats_list: list[dict] = []
    num_dataframes: int = len(workload_paths)

    for i in range(num_dataframes):
        available_trials = get_available_trials(workload_paths[i], depth)
        dataframe, stats = dataframe_from_trials(available_trials)
        df_list.append(dataframe)
        stats_list.append(stats)

    return df_list, stats_list


def get_output_filename(workloads: list[Path]) -> tuple[str, str]:
    some_workload = workloads[0]
    # TODO dynamic naming for multiple dirs? maybe take parser arg of "workflow" and only numerate submissions
    # we could also get this info out of args_file, but i only realized this after coding the directory extracting
    task = Path(some_workload).resolve()
    optimizer = Path(task).parent
    if args.verbose:
        print(f"{task.name=}")
        print(f"{optimizer.name=}")
    file_type = DEFAULT_FILE_ENDING if not args.pdf else "pdf"
    output_filename = f"{optimizer.name}-{task.name}"
    output_filename = here / output_filename
    if args.output:
        output_filename = args.output
    return output_filename, file_type


def set_plotstyle():
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"  # You can adjust the font family as needed
    plt.rcParams["font.size"] = 8  # Adjust the font size as needed


def save_csv(do_save: bool, dfs: list[pd.DataFrame], output_filename: str, verbose: bool):
    if not do_save:
        return
    for i, df in enumerate(dfs):
        csv_output_filename = f"{output_filename}-{i}.csv"
        if verbose:
            print(f"saving raw data as {csv_output_filename}")
        df.to_csv(path_or_buf=csv_output_filename, index=False)


def save_plot(fig, axs, output_filename: str, file_type: str):
    plot_output_filename = f"{output_filename}-heatmap.{file_type}"
    if args.verbose:
        print(f"saving figure as {plot_output_filename}")
    plt.savefig(plot_output_filename)


def main(args: argparse.Namespace):
    workloads: List[Path] = args.trials_dirs
    if args.verbose:
        print(f"{workloads}=")

    output_filename, file_type = get_output_filename(workloads)

    set_plotstyle()

    dfs, stats = extract_dataframes(workloads, depth=args.depth)
    fig, axs = create_figure(dfs, stats)

    save_csv(args.csv, dfs, output_filename, args.verbose)
    save_plot(fig, axs, output_filename, file_type)


if __name__ == "__main__":
    # default paths
    here = Path(__file__).parent.resolve()
    trials_dirs_default = [here / "sample-data" / "test-submission" / "test-workload"]

    # parsing
    parser = argparse.ArgumentParser(description="Create a heatmap plot of benchmarking results.")
    parser.add_argument("--trials_dirs", "-d", default=trials_dirs_default, required=False, nargs='+', type=Path,
                        help="Path to the experiment files (data to plot)")
    parser.add_argument("--depth", default=1, type=int,
                        help="the depth of the trial dirs relative to the given trial_dirs")
    parser.add_argument("--metric", "-m", default="test_acc", required=False, type=str,
                        help="name of metric that should be extracted from result json.")
    parser.add_argument("--x_axis", "-x", required=False, type=str, default="weight_decay",
                        help="parameter for x-axis of the heatmap.")
    parser.add_argument("--y_axis", "-y", required=False, type=str, default="learning_rate",
                        help="parameter for y-axis of the heatmap.")
    parser.add_argument("--output", "-o", required=False, type=Path,
                        help="Filename of the generated output plot. default is *here*.")
    parser.add_argument("--pdf", action="store_true",
                        help="create a pdf instead of a png file")
    parser.add_argument("--limits", required=False, type=int, nargs=2,
                        help="sets the limits for the colormap, 2 ints, order does not matter")
    parser.add_argument("--scale", default=1.0, type=float,
                        help="scales *figsize* argument by this value")
    parser.add_argument("--format", default="2.3", type=str,
                        help="how many digits to display, expects a value seperated by a dot (e.g. 2.3):\
                            multiply by 10^2 and display 3 digits after decimal point. 2.0 for percent")
    parser.add_argument("--last_instead_of_best", "-l", action="store_true",
                        help="use the final model instead of the best one for the plot")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="include debug prints")
    parser.add_argument("--std", action="store_true",
                        help="include standard deviation")
    parser.add_argument("--csv", action="store_true",
                        help="additionaly save data as csv")

    # parser.add_argument("--submission", "-s", required=True, type=Path, help="")
    # parser.add_argument("--workload", "-w", required=True, type=Path, help="")
    args = parser.parse_args()

    main(args)
