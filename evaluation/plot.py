import json
from sys import exit
from pathlib import Path
from os import PathLike
from typing import List, Literal
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_warn
from engine.parser import YAMLParser
from engine.utils import AttributeDict, convert_type_inside_dict
from evaluation import evaluation_path
from itertools import repeat


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


def dataframe_from_trials(trial_dir_paths: List[Path], config: AttributeDict) -> pd.DataFrame:
    """takes result from get_available_trials and packs them in a dataframe,
    does not filter duplicate hyperparameter settings."""
    dfs: List[pd.DataFrame] = []

    for path in trial_dir_paths:

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
            print(f"  <{config_file}>: {config_file.is_file()} and\n  <{result_file}>: {result_file.is_file()})")
            continue

        yaml_parser = YAMLParser()
        yaml_content = yaml_parser.parse_yaml(config_file)
        # convert the sub dicts first, then the dict itself
        yaml_content = convert_type_inside_dict(yaml_content, src=dict, tgt=AttributeDict)
        yaml_content = AttributeDict(yaml_content)
        if config.verbose:
            print(f"{yaml_content=}\n")

        # use user given value
        metric_of_value_to_plot = config.plot.metric

        # compute it if user has not given a value
        if not metric_of_value_to_plot:
            raise ValueError("evaluation.plot.metric is not set")

        data = pd.json_normalize(yaml_content)

        with open(result_file, "r", encoding="utf8") as f:
            content = json.load(f)
            if metric_of_value_to_plot in content[0]:
                data.at[0, metric_of_value_to_plot] = content[0][metric_of_value_to_plot]
            else:
                rank_zero_warn(f"could not find value for {metric_of_value_to_plot} in json")

        dfs.append(data)

    if len(dfs) == 0:
        raise ValueError("no dataframes found, check your config")
    df = pd.concat(dfs, sort=False)

    return df


def create_matrix_plot(dataframe, config: AttributeDict, cols: str, idx: str, ax=None, low_is_better: bool = False,
                       cbar: bool = True, vmin: None | int = None, vmax: None | int = None):
    """
    Creates one heatmap and puts it into the grid of subplots.
    Uses pd.pivot_table() and sns.heatmap().
    """
    df_entry = dataframe.iloc[0]
    task_name = df_entry["task.name"]
    metric_name = df_entry["evaluation.plot.metric"]

    # CLEANING LAZY USER INPUT
    # cols are x-axis, idx are y-axis
    if cols not in dataframe.columns:
        print("Warning: x-axis value not present in the dataframe; did you forget to add a 'optimizer.' as a prefix?\n" +
              f"  using '{'optimizer.' + cols}' as 'x-axis' instead.")
        cols = "optimizer." + cols
    if idx not in dataframe.columns:
        print("Warning: y-axis value not present in the dataframe; did you forget to add a 'optimizer.' as a prefix?\n" +
              f"  using '{'optimizer.' + idx}' as 'y-axis' instead.")
        idx = "optimizer." + idx
    # create pivot table and format the score result
    pivot_table = pd.pivot_table(dataframe,
                                 columns=cols, index=idx, values=metric_name,
                                 aggfunc='mean')

    fmt = None
    format_string = config.plot.format

    # scaline the values given by the user to fit his format needs (-> and adapting the limits)
    value_exp_factor, decimal_points = format_string.split(".")
    value_exp_factor = int(value_exp_factor)
    decimal_points = int(decimal_points)
    if vmin:
        vmin *= (10 ** value_exp_factor)
    if vmax:
        vmax *= (10 ** value_exp_factor)
    pivot_table = (pivot_table * (10 ** value_exp_factor)).round(decimal_points)
    fmt=f".{decimal_points}f"

    vmin = min(config.plot.limits)
    vmax = max(config.plot.limits)

    if config.verbose:
        print(f"setting cbar limits to {vmin}, {vmax} ")
        print(pivot_table)

    colormap_name = config.plotstyle.color_palette
    if low_is_better:
        colormap_name += "_r"  # this will "inver" / "flip" the colorbar
    colormap = sns.color_palette(colormap_name, as_cmap=True)
    metric_legend = pretty_name(metric_name, config)

    # FINETUNE POSITION
    # left bottom width height
    # cbar_ax = fig.add_axes([0.92, 0.235, 0.02, 0.6])
    cbar_ax = None

    if not config.plot.std:
        return sns.heatmap(pivot_table, ax=ax, cbar_ax=cbar_ax,
                           annot=True, fmt=fmt,
                           annot_kws={'fontsize': config.plotstyle.matrix_font.size},
                           cbar=cbar, vmin=vmin, vmax=vmax, cmap=colormap, cbar_kws={'label': f"{metric_legend}"})
    else:
        # BUILD STD TABLE
        pivot_table_std = pd.pivot_table(dataframe,
                                        columns=cols, index=idx, values=metric_name,
                                        aggfunc=config.plot.aggfunc,  fill_value=float("inf"), dropna=False
                                        )
        if float("inf") in pivot_table_std.values.flatten():
            print("WARNING: Not enough data to calculate the std, skipping std in plot")

        pivot_table_std = (pivot_table_std * (10 ** value_exp_factor)).round(decimal_points)

        annot_matrix = pivot_table.copy().astype("string")
        for i in pivot_table.index:
            for j in pivot_table.columns:
                mean = pivot_table.loc[i, j]
                std = pivot_table_std.loc[i, j]
                std_string = f"\n±({round(std, decimal_points)})" if std != float("inf") else ""  # type: ignore
                annot_matrix.loc[i, j] = f"{round(mean, decimal_points)}{std_string}"  # type: ignore

        fmt = ""  # cannot format like before, as we do not only have a number

        return sns.heatmap(pivot_table, ax=ax, cbar_ax=cbar_ax,
                           annot=annot_matrix, fmt=fmt,
                           annot_kws={'fontsize': config.plotstyle.matrix_font.size},
                           cbar=cbar, vmin=vmin, vmax=vmax, cmap=colormap, cbar_kws={'label': f"{metric_legend}"})


def get_all_num_rows_and_their_names(dataframe_list: list[pd.DataFrame], config):
    n_rows: list[int] = []
    row_names: list[list[str]] = []
    for i, df in enumerate(dataframe_list):
        seed = "seed"  # legacy
        engine_seed = "engine.seed"
        x_axis = config.plot.x_axis[i]
        y_axis = config.plot.y_axis[0]
        if df["evaluation.plot.metric"].nunique() > 1:
            rank_zero_warn("More than one metric found, using the first one.")
        metric = df["evaluation.plot.metric"].unique()[0]
        ignored_cols = [x_axis, y_axis, engine_seed, seed, metric]
        ignored_cols += config.get("ignore_keys", [])
        current_n_rows, current_names = get_num_rows(df, ignored_cols, config)
        n_rows.append(current_n_rows)
        if not current_names:  # will be empty if we have only one row
            current_names.append("default")
        row_names.append(current_names)

    return n_rows, row_names

def get_num_rows(dataframe: pd.DataFrame, ignored_cols: list[str], config: AttributeDict
                 ) -> tuple[int, list[str]]:
    """each matrix has 2 params (on for x and y each), one value, and we aggregate over seeds;
    if there are more than than these 4 parameter with different values,
    we want to put that in seperate rows instead of aggregating over them.
    returning: the number of rows (atleast 1) and the names of the cols"""
    necesarry_rows = 0

    # the user might specify a value for the groups that we should split on in <split_groups>
    whitelisted_cols: list[str] | Literal["all", ""] = "all"  # everything is whitelisted if this value stays N this value if it stays None
    if isinstance(config.split_groups, str):
        # user has given one value, we want to process a list
        whitelisted_cols = [config.split_groups]
    elif isinstance(config.split_groups, list):
        whitelisted_cols = config.split_groups[:]
    elif config.split_groups == False:
        whitelisted_cols = ""

    columns_with_non_unique_values = []
    for col in dataframe.columns:
        is_eval_key = col.startswith("evaluation.")
        is_ignored = col in ignored_cols
        is_whitelisted = whitelisted_cols == "all" or col in whitelisted_cols
        if is_ignored or is_eval_key or not is_whitelisted:
            if config.verbose:
                print(f"ignoring {col}")
            continue
        nunique = dataframe[col].nunique(dropna=False)
        if nunique > 1:
            if config.verbose:
                print(f"adding {col} since there are {nunique} unique values")
            for unique_hp in dataframe[col].unique():
                columns_with_non_unique_values.append(f"{col}={unique_hp}")
            necesarry_rows += (nunique)  # each unique parameter should be an indivudal plot

    rows_number = max(necesarry_rows, 1)
    col_names = columns_with_non_unique_values
    if config.verbose:
        print(f"{rows_number=}")
        print(f"{col_names=}")

    return rows_number, col_names


def find_global_vmin_vmax(dataframe_list, config):
    vmin: int | None = None
    vmax: int | None = None
    num_cols = len(dataframe_list)

    if num_cols > 1:
        # all subplots should have same colors -> we need to find the limits
        vmin = float('inf')
        vmax = float('-inf')
        if config.verbose:
            print(f"===== cbar limits =====")
            print()
        for i in range(num_cols):
            dataframe = dataframe_list[i]
            cols = config.plot.x_axis[i]
            idx = config.plot.y_axis[0]
            key = config.plot.metric

            pivot_table = pd.pivot_table(dataframe,
                                 columns=cols, index=idx,
                                 values=key,
                                 aggfunc='mean')

            min_value_present_in_current_df = pivot_table.min().min()
            max_value_present_in_current_df = pivot_table.max().max()

            if config.verbose:
                print(f"subfigure number {i+1}, checking for metric {key}: \n" +
                    f" min value is {min_value_present_in_current_df},\n" +
                    f" max value is {max_value_present_in_current_df},\n")
            vmin = min(vmin, min_value_present_in_current_df)
            vmax = max(vmax, max_value_present_in_current_df)

    return vmin, vmax


def create_figure(dataframe_list: list[pd.DataFrame], config: AttributeDict):
    """
    Takes a list of dataframes. Each dataframe is processed into a column of heatmaps.
    """
    num_cols: int = len(dataframe_list)

    # calculate the number of rows for each dataframe
    n_rows, row_names = get_all_num_rows_and_their_names(dataframe_list, config)

    # Handling of the number of rows in the plot
    # we could either create a full rectangular grid, or allow each subplot to nest subplots
    # for nesting we would need to create subfigures instead of subplots i think
    if config.split_groups == False:
        n_rows_max = 1
        row_names = [["default"] for _ in range(num_cols)]
    else:
        n_rows_max = max(n_rows)

    if config.verbose:
        print(f"{n_rows=}")
        print(f"{num_cols=}")

    # TODO, figsize was just hardcoded for (1, 2) grid and left to default for (1, 1) grid
    #       probably not worth the hazzle to create something dynamic (atleast not now)
    # EDIT: it was slightly adapted to allow num rows without being completely unreadable
    # margin = (num_subfigures - 1) * 0.3
    # figsize=(5*n_cols + margin, 2.5)
    figsize = None
    if num_cols == 2:
        # TODO: after removing cbar from left subifgure, it is squished
        #       there is an argument to share the legend, we should use that
        figsize = (12 * config.plotstyle.scale, 5.4 * n_rows_max * config.plotstyle.scale)
    elif num_cols > 2:
        figsize = (12 * (num_cols / 2) * config.plotstyle.scale, 5.4 * n_rows_max * config.plotstyle.scale)

    fig, axs = plt.subplots(n_rows_max, num_cols, figsize=figsize)
    if num_cols == 1:
        axs = [axs]  # adapt for special case so we have unified types
    if n_rows_max == 1:
        axs = [axs]

    # Adjust left and right margins as needed
    # fig.subplots_adjust(left=0.1, right=0.9, top=0.97, hspace=0.38, bottom=0.05,wspace=0.3)

    # None -> plt will chose vmin and vmax
    vmin, vmax = find_global_vmin_vmax(dataframe_list, config)

    for i in range(num_cols):
        num_nested_subfigures: int = n_rows[i]

        if not config.split_groups:
            create_one_grid_element(dataframe_list, config, axs, i,
                                    j=0,
                                    max_i=num_cols,
                                    max_j=0,
                                    vmin=vmin,
                                    vmax=vmax,
                                    n_rows=n_rows,
                                    row_names=row_names)
        else:
            for j in range(num_nested_subfigures):
                create_one_grid_element(dataframe_list, config, axs, i,
                                        j,
                                        max_i=num_cols,
                                        max_j=num_nested_subfigures,
                                        vmin=vmin,
                                        vmax=vmax,
                                        n_rows=n_rows,
                                        row_names=row_names)

    if config.plotstyle.tight_layout:
        fig.tight_layout()
    # SUPTITLE (the super title on top of the whole figure in the middle)
    if n_rows_max > 1 or num_cols > 1:
        # set experiment name as title when multiple matrices in image
        # TODO super title might be squished when used together with tight layout
        if config.experiment_name:
            fig.suptitle(config.experiment_name)
    return fig, axs


def create_one_grid_element(dataframe_list: list[pd.DataFrame], config: AttributeDict, axs,
                            i: int, j: int, max_i: int, max_j: int, vmin, vmax, n_rows, row_names):
    """does one 'axs' element as it is called in plt"""
    num_nested_subfigures: int = n_rows[i]
    name_for_additional_subplots: list[str] = row_names[i]
    num_subfigures = max_i  # from left to right
    num_nested_subfigures = max_j  # from top to bottom
    dataframe = dataframe_list[i]
    df_entry = dataframe.iloc[0]  # just get an arbitrary trial for the target metric mode and submission name
    opti_name = df_entry['optimizer.name']
    low_is_better = config.plot.test_metric_mode == "min"

    cols = config.plot.x_axis[i]
    idx = config.plot.y_axis[0]
    # only include colorbar once
    include_cbar: bool = i == num_subfigures - 1

    model_param = name_for_additional_subplots[j]
    if model_param == "default":
        current_dataframe = dataframe  # we do not need to do further grouping
    else:
        param_name, param_value = model_param.split("=", maxsplit=1)
        if pd.api.types.is_numeric_dtype(dataframe[param_name]):
            param_value = float(param_value)
        try:
            current_dataframe = dataframe.groupby([param_name]).get_group(param_value)
        except KeyError:
            if config.verbose:
                print(f"{param_name=}")
                print(f"{param_value=}")
                print(f"{dataframe.columns=}")
                print(f"{dataframe[param_name]=}")
            print(f"WARNING: was not able to groupby '{param_name}'," +
                    "maybe the data was created with different versions of fob; skipping this row")
            return False
    current_plot = create_matrix_plot(current_dataframe, config,
                                        cols, idx,
                                        ax=axs[j][i], low_is_better=low_is_better,
                                        cbar=include_cbar, vmin=vmin, vmax=vmax)

    # LABELS
    # Pretty name for label "learning_rate" => "Learning Rate"
    # remove x_label of all but last row, remove y_label for all but first column    
    if i > 0:
        current_plot.set_ylabel('', labelpad=8)
    else:
        current_plot.set_ylabel(pretty_name(current_plot.get_ylabel(), config))
    if j < num_nested_subfigures - 1:
        current_plot.set_xlabel('', labelpad=8)
    else:
        current_plot.set_xlabel(pretty_name(current_plot.get_xlabel(), config))

    # TITLE
    # title (heading) of the heatmap <optimname> on <taskname> (+ additional info)
    title = pretty_name(opti_name, config)
    title += " on "
    title += pretty_name(df_entry["task.name"], config)
    if max_i > 1 or max_j > 1:
        title += "" if model_param == "default" else f"\n{model_param}"
    current_plot.set_title(title)


def extract_dataframes(workload_paths: List[Path], config: AttributeDict, depth: int = 1
                       ) -> list[pd.DataFrame]:
    df_list: list[pd.DataFrame] = []
    num_dataframes: int = len(workload_paths)

    for i in range(num_dataframes):
        available_trials = get_available_trials(workload_paths[i], config, depth)
        dataframe = dataframe_from_trials(available_trials, config)
        df_list.append(dataframe)

    return df_list


def get_output_file_path(dataframe_list: list[pd.DataFrame], config: AttributeDict, suffix: str = "") -> Path:
    task_names = [df.iloc[0]["task.name"] for df in dataframe_list]
    optim_names = [df.iloc[0]["optimizer.name"] for df in dataframe_list]
    task_name = "_".join(task_names)
    optim_name = "_".join(optim_names)

    if config.verbose:
        print(f"{task_name=}")
        print(f"{task_name=}")

    here = Path(__file__).parent.resolve()

    output_dir = Path(config.output_dir) if config.output_dir else here
    experiment_name = Path(config.experiment_name) if config.experiment_name else f"{optim_name}-{task_name}"
    output_file_path = output_dir / experiment_name

    return Path(f"{output_file_path}-{suffix}" if suffix else output_file_path)


def set_plotstyle(config: AttributeDict):
    plt.rcParams["text.usetex"] = config.plotstyle.text.usetex
    plt.rcParams["font.family"] = config.plotstyle.font.family
    plt.rcParams["font.size"] = config.plotstyle.font.size


def pretty_name(name: str, config: AttributeDict, pretty_names: dict | str = {}) -> str:
    """tries to use a mapping for the name, else will do some general replacement.
    mapping can be a directory or a filename of a yaml file with 'names' key"""
    
    # reading from yaml and caching the dictionary
    label_file: Path = evaluation_path() / "labels.yaml"
    if isinstance(pretty_names, str):
        label_file = Path(pretty_names)
    
    if pretty_names == {} or isinstance(pretty_names, str):
        yaml_parser = YAMLParser()
        yaml_content = yaml_parser.parse_yaml(label_file)
        pretty_names = yaml_content["names"]

    # applying pretty names
    name_without_yaml_prefix = name.split(".")[-1]
    if name in pretty_names.keys():
        name = pretty_names[name]
    elif name_without_yaml_prefix in pretty_names.keys():
        name = pretty_names[name_without_yaml_prefix]
    else:
        name = name.replace('_', ' ').title()
    return name


def save_csv(dfs: list[pd.DataFrame], output_filename: Path, verbose: bool):
    for i, df in enumerate(dfs):
        csv_output_filename = f"{output_filename.resolve()}-{i}.csv"
        if verbose:
            print(f"saving raw data as {csv_output_filename}")
        df.to_csv(path_or_buf=csv_output_filename, index=False)


def save_plot(output_file_path: Path, file_type: str, verbose: bool):
    plot_output_filename = f"{output_file_path.resolve()}-heatmap.{file_type}"
    if verbose:
        print(f"saving figure as {plot_output_filename}")
    plt.savefig(plot_output_filename)


def save_files(dfs: list[pd.DataFrame], output_file_path: Path, config: AttributeDict):
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    for file_type in config.output_types:
        if file_type == "csv":
            save_csv(dfs, output_file_path, config.verbose)
        elif file_type == "png" or file_type == "pdf":
            save_plot(output_file_path, file_type, config.verbose)
    full_path = output_file_path.resolve()
    print(f"Saved results into <{full_path}>")


def clean_config(config: AttributeDict) -> AttributeDict:
    """some processing that allows the user to be lazy, shortcut for the namespace, hidden values are found and config.all_values"""
    # print(f"{config=}")
    if "evaluation" in config.keys():
        evaluation_config: AttributeDict = config.evaluation
        evaluation_config["all_values"] = config
        # print(f"{evaluation_config=}")
        config = evaluation_config
    else:
        print("WARNING: there is no 'evaluation' in the yaml provided!")
    if "data_dirs" in config.keys():
        value_is_none = not config.data_dirs
        value_has_wrong_type = not isinstance(config.data_dirs, (PathLike, str, list))
        if value_is_none or value_has_wrong_type:
            exit(f"Error: 'evaluation.data_dirs' was not provided correctly! check for typos in the yaml provided! value given: {config.data_dirs}")


    # print(f"{config=}")
    # allow the user to write a single string instead of a list of strings
    if not isinstance(config.output_types, list):
        config["output_types"] = [config.output_types]
        if config.verbose:
            print("fixing value for key <config.output_types> to be a list[str]")

    if not isinstance(config.data_dirs, list):
        config["data_dirs"] = [Path(config.data_dirs)]
        if config.verbose:
            print("fixing value for key <config.data_dirs> to be a list[Path]")

    # x_axis
    if not isinstance(config.plot.x_axis, list):
        config["plot"]["x_axis"] = [config.plot.x_axis]
        if config.verbose:
            print("fixing value for key <config.plot.x_axis> to be a list[str]")
    if len(config.plot.x_axis) < len(config.data_dirs):
        # use same x axis for all if only one given
        missing_elements = len(config.data_dirs) - len(config.plot.x_axis)
        config["plot"]["x_axis"] += repeat(config.plot.x_axis[0], missing_elements)

    # y_axis
    if not isinstance(config.plot.y_axis, list):
        config["plot"]["y_axis"] = [config.plot.y_axis]
        if config.verbose:
            print("fixing value for key <config.plot.y_axis> to be a list[str]")
    if len(config.plot.y_axis) < len(config.data_dirs):
        # use same x axis for all if only one given
        missing_elements = len(config.data_dirs) - len(config.plot.y_axis)
        config["plot"]["y_axis"] += repeat(config.plot.y_axis[0], missing_elements)

    return config


def main(config: AttributeDict):
    config = clean_config(config)  # sets config to config.evaluation, cleans some data
    workloads: List[Path] = [Path(name) for name in config.data_dirs]
    if config.verbose:
        print(f"{workloads}=")

    set_plotstyle(config)

    dfs = extract_dataframes(workloads, depth=config.depth, config=config)
    fig, axs = create_figure(dfs, config)

    output_file_path = get_output_file_path(dfs, config)

    
