import json
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from engine.parser import YAMLParser
from engine.utils import AttributeDict, convert_type_inside_dict


def get_available_trials(dirname: Path, config: AttributeDict, depth: int = 1):
    """finds the path for all trials in the *dirname* directory"""
    # RECURSIVELY FIND ALL DIRS IN DIRNAME (up to depth)
    assert isinstance(dirname, Path)
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

    if config.verbose:
        format_string = "\n  "
        print(f"found the following directories:{format_string}{format_string.join(str(i) for i in subdirs)}.")

    def is_trial(path: Path):
        # here we could do additional checks to filter the subdirectories
        # currently we only check if there is a config file
        for x in path.iterdir():
            found_a_config_file = x.name == config.experiment_files.config
            if found_a_config_file:
                return True
        return False

    subdirs = list(filter(is_trial, subdirs[::-1]))
    if config.verbose:
        print(f"we assume the following to be trials:{format_string}{format_string.join(str(i) for i in subdirs)}.")
    return subdirs


def dataframe_from_trials(trial_dir_paths: List[Path], config: AttributeDict):
    """takes result from get_available_trials and packs them in a dataframe,
    does not filter duplicate hyperparameter settings."""
    dfs: List[pd.DataFrame] = []
    stats: list[dict] = []

    for path in trial_dir_paths:
        stat = {}
        hp_dict = {}

        config_file = path / config.experiment_files.config
        if config.last_instead_of_best:
            result_file = path / config.experiment_files.last_model
        else:
            result_file = path / config.experiment_files.best_model
        all_files_exist = all([
            config_file.is_file(),
            result_file.is_file()
        ])
        if not all_files_exist:
            print(f"WARNING: one or more files are missing in {path}. Skipping this hyperparameter setting.")
            continue

        yaml_parser = YAMLParser()
        yaml_content = yaml_parser.parse_yaml(config_file)
        # convert the sub dicts first, then the dict itself
        yaml_content = convert_type_inside_dict(yaml_content, src=dict, tgt=AttributeDict)
        yaml_content = AttributeDict(yaml_content)
        if config.verbose:
            print(f"{yaml_content=}\n")

        stat["seed"] = yaml_content.engine.seed
        stat["task_name"] = yaml_content.task.name
        stat["optimizer_name"] = yaml_content["optimizer"]["name"]
        stat["target_metric_mode"] = yaml_content["task"]["target_metric_mode"]
        stat["target_metric"] = yaml_content["task"]["target_metric"]
        # TARGET_METRIC_MODE = stat["target_metric_mode"]  # max or min
        hp_dict = yaml_content["optimizer"]

        # print(f"{type(config.plot.metric)=}")
        metric_of_value_to_plot = config.plot.metric
        if not metric_of_value_to_plot:
            task_name = stat["task_name"]
            metric_of_value_to_plot = config.task_to_metric[task_name]
            if not metric_of_value_to_plot:
                metric_of_value_to_plot = stat["target_metric"].replace("val_", "test_")
                print(f"WARNING: no metric given'... " +
                      f"Using '{metric_of_value_to_plot}' because '{stat['target_metric']}' was the target metric.")
        stat['metric'] = metric_of_value_to_plot

        with open(result_file) as f:
            content = json.load(f)
            if metric_of_value_to_plot in content[0]:
                stat["score"] = content[0][metric_of_value_to_plot]
            else:
                stat["score"] = f"could not find value for {metric_of_value_to_plot} in json"

        data = pd.json_normalize(hp_dict)
        data.at[0, metric_of_value_to_plot] = stat["score"]  # will trim to 4 digits after comma
        data.at[0, "seed"] = stat["seed"]  # saved as float
        dfs.append(data)  # append the data frame to the list

        if config.verbose:
            print(f"{stat=}")
        stats.append(stat)

    df = pd.concat(dfs, sort=False)

    return df, stats


def create_matrix_plot(dataframe, config: AttributeDict, ax=None, low_is_better: bool = False, stat: dict = {}):
    # create pivot table and format the score result
    pivot_table = pd.pivot_table(dataframe,
                                 columns=config.plot.x_axis, index=config.plot.y_axis, values=stat["metric"],
                                 aggfunc='mean')
    value_exp_factor, decimal_points = config.plot.format.split(".")
    value_exp_factor = int(value_exp_factor)
    decimal_points = int(decimal_points)
    pivot_table = (pivot_table * (10 ** value_exp_factor)).round(decimal_points)
    if config.verbose:
        print(pivot_table)

    # setting the COLORMAP and the RANGE of the values for the colors (legend bar)
    vmin = config.plot.limits and min(config.plot.limits)  # lower limit (or None if not given)
    vmax = config.plot.limits and max(config.plot.limits)  # upper limit (or None if not given)
    colormap_name = "rocket"
    if low_is_better:
        colormap_name += "_r"  # this will "inver" / "flip" the colorbar
    colormap = sns.color_palette(colormap_name, as_cmap=True)
    # metric_legend = stat["metric"] if "metric" in stat.keys() else config.plot.metric
    metric_legend = stat["metric"]
    metric_legend = pretty_name(metric_legend, config)

    # FINETUNE POSITION
    # left bottom width height
    # cbar_ax = fig.add_axes([0.92, 0.235, 0.02, 0.6])
    cbar_ax = None

    if not config.plot.std:
        return sns.heatmap(pivot_table, ax=ax, cbar_ax=cbar_ax,
                           annot=True, fmt=f".{decimal_points}f",
                           annot_kws={'fontsize': config.plotstyle.matrix_font.size},
                           vmin=vmin, vmax=vmax, cmap=colormap, cbar_kws={'label': f"{metric_legend}"})
    else:
        # PRECISION TO DISPLAY
        std_decimal_points = decimal_points  # good default would be 2
        mean_decimal_points = decimal_points  # good default would be 1

        # BUILD STD TABLE
        pivot_table_std = pd.pivot_table(dataframe,
                                         columns=config.plot.x_axis, index=config.plot.y_axis, values=stat["metric"],
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
                           annot=annot_matrix, fmt=fmt,
                           annot_kws={'fontsize': config.plotstyle.matrix_font.size},
                           vmin=vmin, vmax=vmax, cmap=colormap, cbar_kws={'label': f"{metric_legend}"})


def create_figure(dataframe_list: list[pd.DataFrame], stats_list: list[dict], config: AttributeDict):
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
        figsize = (12 * config.plotstyle.scale, 5.4 * config.plotstyle.scale)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    if num_subfigures == 1:
        axs = [axs]  # adapt for special case so we have unified types

    # Adjust left and right margins as needed
    # fig.subplots_adjust(left=0.1, right=0.9, top=0.97, hspace=0.38, bottom=0.05,wspace=0.3)
    for i in range(num_subfigures):
        dataframe, stats = dataframe_list[i], stats_list[i]
        stat_entry = stats[0]  # just get an arbitrary trial for the target metric mode and submission name
        opti_name = stat_entry['optimizer_name']
        s_target_metric_mode = stat_entry["target_metric_mode"]
        low_is_better = s_target_metric_mode == "min"

        current_plot = create_matrix_plot(dataframe, config, ax=axs[i], low_is_better=low_is_better, stat=stat_entry)

        # Pretty name for label "learning_rate" => "Learning Rate"
        current_plot.set_xlabel(pretty_name(current_plot.get_xlabel(), config))
        current_plot.set_ylabel(pretty_name(current_plot.get_ylabel(), config))

        if i > 0:
            # remove y_label of all but first one
            # axs[i].set_ylabel('', fontsize=8, labelpad=8)
            axs[i].set_ylabel('', labelpad=8)
        else:
            # TODO format parameter just as in submission name
            # axs[i].set_ylabel
            pass

        # title (heading) of the figure:
        title = pretty_name(opti_name, config)
        title += " on "
        title += pretty_name(stat_entry["task_name"], config)
        axs[i].set_title(title)

    if config.plotstyle.tight_layout:
        fig.tight_layout()
    if len(config.data_dirs) > 1:
        # set experiment name as title when multiple matrices in image
        # super title TODO fix when used together with tight layout
        # print(config.experiment_name)
        if name := config.experiment_name:
            fig.suptitle(name)
    return fig, axs


def extract_dataframes(workload_paths: List[Path], config: AttributeDict, depth: int = 1
                       ) -> tuple[list[pd.DataFrame], list[dict]]:
    df_list: list[pd.DataFrame] = []
    stats_list: list[dict] = []
    num_dataframes: int = len(workload_paths)

    for i in range(num_dataframes):
        available_trials = get_available_trials(workload_paths[i], config, depth)
        dataframe, stats = dataframe_from_trials(available_trials, config)
        df_list.append(dataframe)
        stats_list.append(stats)

    return df_list, stats_list


def get_output_file_path(workloads: list[Path], config: AttributeDict, stats: list[dict]) -> str:
    some_workload = workloads[0]
    some_stat = stats[0][0]  # TODO: fix for multiple optim on one plot
    # TODO dynamic naming for multiple dirs? maybe take parser arg of "workflow" and only numerate submissions
    # we could also get this info out of args_file, but i only realized this after coding the directory extracting
    # optimizer = Path(some_workload).resolve()
    # optim_name = optimizer.name
    # task = Path(optimizer).parent
    # task_name = task.name
    # print(f"{stats=}")
    # print(f"{some_stat=}")
    task_name = some_stat["task_name"]
    optim_name = some_stat["optimizer_name"]

    if config.verbose:
        print(f"{task_name=}")
        print(f"{task_name=}")

    here = Path(__file__).parent.resolve()

    output_dir = Path(config.output_dir) if config.output_dir else here
    experiment_name = Path(config.experiment_name) if config.experiment_name else f"{optim_name}-{task_name}"
    output_file_path = output_dir / experiment_name

    return output_file_path


def set_plotstyle(config: AttributeDict):
    plt.rcParams["text.usetex"] = config.plotstyle.text.usetex
    plt.rcParams["font.family"] = config.plotstyle.font.family
    plt.rcParams["font.size"] = config.plotstyle.font.size


def pretty_name(name: str, config: AttributeDict) -> str:
    """tries to use a mapping for the name, else will do some general replacement"""
    if name in config.names.keys():
        name = config.names[name]
    else:
        name = name.replace('_', ' ').title()
    return name


def save_csv(dfs: list[pd.DataFrame], output_filename: str, verbose: bool):
    for i, df in enumerate(dfs):
        csv_output_filename = f"{output_filename}-{i}.csv"
        if verbose:
            print(f"saving raw data as {csv_output_filename}")
        df.to_csv(path_or_buf=csv_output_filename, index=False)


def save_plot(fig, axs, output_file_path: str, file_type: str, verbose: bool):
    plot_output_filename = f"{output_file_path}-heatmap.{file_type}"
    if verbose:
        print(f"saving figure as {plot_output_filename}")
    plt.savefig(plot_output_filename)


def clean_config(config: AttributeDict) -> AttributeDict:
    # allow the user to write a single string instead of a list of strings
    if not isinstance(config.output_types, list):
        config["output_types"] = [config.output_types]
        if config.verbose:
            print("fixing value for key <config.output_types> to be a list[str]")

    if not isinstance(config.data_dirs, list):
        config["data_dirs"] = [Path(config.data_dirs)]
        if config.verbose:
            print("fixing value for key <config.data_dirs> to be a list[str]")

    # something weird going here, we just cast it again
    config = convert_type_inside_dict(config, dict, AttributeDict)
    config = AttributeDict(config)

    return config


def main(config: AttributeDict):
    config = clean_config(config)
    workloads: List[Path] = [Path(name) for name in config.data_dirs]
    if config.verbose:
        print(f"{workloads}=")

    set_plotstyle(config)

    dfs, stats = extract_dataframes(workloads, depth=config.depth, config=config)
    fig, axs = create_figure(dfs, stats, config)

    output_file_path = get_output_file_path(workloads, config, stats)

    for file_type in config.output_types:
        if file_type == "csv":
            save_csv(dfs, output_file_path, config.verbose)
        elif file_type == "png" or file_type == "pdf":
            save_plot(fig, axs, output_file_path, file_type, config.verbose)
    print(f"Saved results into <{output_file_path}>")
