import json
from dataclasses import dataclass
from itertools import repeat
from os import PathLike
from pathlib import Path
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from pytorch_fob.engine.parser import YAMLParser
from pytorch_fob.engine.utils import AttributeDict, convert_type_inside_dict, log_debug, log_info, log_warn
from pytorch_fob.evaluation import evaluation_path


@dataclass
class PivotTable:
    grouped_df: pd.DataFrame
    pivot_table: pd.DataFrame
    std_table: Optional[pd.DataFrame] = None

    @property
    def empty(self) -> bool:
        return self.pivot_table.empty


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
    format_str = "\n  "  # f-string expression part cannot include a backslash
    log_debug(f"found the following directories:{format_str}{format_str.join(str(i) for i in subdirs)}.")

    def is_trial(path: Path):
        # here we could do additional checks to filter the subdirectories
        # currently we only check if there is a config file
        for x in path.iterdir():
            found_a_config_file = x.name == config.experiment_files.config
            if found_a_config_file:
                return True
        return False

    subdirs = list(filter(is_trial, subdirs[::-1]))
    log_debug(f"We assume the following to be trials:{format_str}{format_str.join(str(i) for i in subdirs)}.")
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
            log_warn(f"WARNING: one or more files are missing in {path}. Skipping this hyperparameter setting." +
                           f"  <{config_file}>: {config_file.is_file()} and\n  <{result_file}>: {result_file.is_file()})")
            continue

        yaml_parser = YAMLParser()
        yaml_content = yaml_parser.parse_yaml(config_file)
        # convert the sub dicts first, then the dict itself
        yaml_content = convert_type_inside_dict(yaml_content, src=dict, tgt=AttributeDict)
        yaml_content = AttributeDict(yaml_content)

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
                log_warn(f"could not find value for {metric_of_value_to_plot} in json")

        dfs.append(data)

    if len(dfs) == 0:
        raise ValueError("no dataframes found, check your config")
    df = pd.concat(dfs, sort=False)

    return df


def get_ticklabels(values, usetex: bool, use_log: bool) -> list[str] | Literal["auto"]:
    def to_loglabel(x: float) -> str:
        if usetex:
            return f"$10^{{{x}}}$"
        else:
            return f"10^({x})"
    if use_log:
        return [to_loglabel(round(np.log10(v), 1)) for v in values]
    else:
        return "auto"
    

def make_colorbar(dataframe: pd.DataFrame, config: AttributeDict, ax, vmin: float | None, vmax: float | None):
    colormap_name = config.plotstyle.color_palette
    low_is_better = dataframe["evaluation.plot.test_metric_mode"].iloc[0] == "min"
    if low_is_better:
        colormap_name += "_r"  # this will "inver" / "flip" the colorbar
    colormap = sns.color_palette(colormap_name, as_cmap=True)
    metric_name = dataframe.iloc[0]["evaluation.plot.metric"]
    metric_legend = pretty_name(metric_name)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=ax, cmap=colormap, label=metric_legend)

def get_exp_dec_format(pt_object: PivotTable) -> tuple[int, int]:
    dataframe = pt_object.grouped_df
    format_string = dataframe["evaluation.plot.format"].iloc[0]
    exponent, decimal_points = format_string.split(".")
    exponent = int(exponent)
    decimal_points = int(decimal_points)
    return exponent, decimal_points


def handle_formatting(pt_object: PivotTable, vmin: float | None, vmax: float | None) -> tuple[float | None, float | None]:
    """adjust vmin, vmax and pivot tables based on the format string in the dataframe"""
    exponent, decimal_points = get_exp_dec_format(pt_object)

    # adjust vmin and vmax to match the formatting
    if vmin is not None:
        vmin *= (10 ** exponent)
    if vmax is not None:
        vmax *= (10 ** exponent)

    # adjust pivot tables to match the formatting
    pt_object.pivot_table = (pt_object.pivot_table * (10 ** exponent)).round(decimal_points)
    if pt_object.std_table is not None:
        pt_object.std_table = (pt_object.std_table * (10 ** exponent)).round(decimal_points)

    return vmin, vmax


def create_matrix_plot(pt_object: PivotTable,
                       config: AttributeDict,
                       cols: str, idx: str, ax=None,
                       vmin: None | float = None, vmax: None | float = None,
                       xticks_log: bool = False, yticks_log: bool = False):
    """
    Creates one heatmap and puts it into the grid of subplots. Uses sns.heatmap().
    """
    dataframe = pt_object.grouped_df
    pivot_table = pt_object.pivot_table

    # CLEANING LAZY USER INPUT
    # cols are x-axis, idx are y-axis
    if cols not in dataframe.columns:
        log_warn("x-axis value not present in the dataframe; did you forget to add a 'optimizer.' as a prefix?\n" +
                       f"  using '{'optimizer.' + cols}' as 'x-axis' instead.")
        cols = "optimizer." + cols
    if idx not in dataframe.columns:
        log_warn("y-axis value not present in the dataframe; did you forget to add a 'optimizer.' as a prefix?\n" +
                       f"  using '{'optimizer.' + idx}' as 'y-axis' instead.")
        idx = "optimizer." + idx

    usetex = config.plotstyle.text.usetex
    _, decimals = get_exp_dec_format(pt_object)

    if pt_object.std_table is not None:
        pivot_table_std = pt_object.std_table
        if float("inf") in pivot_table_std.values.flatten():
            log_warn("WARNING: Not enough data to calculate the std, skipping std in plot")

        # create annotation matrix
        annot_matrix = pivot_table.copy().astype("string")
        for i in pivot_table.index:
            for j in pivot_table.columns:
                mean = pivot_table.loc[i, j]
                std = pivot_table_std.loc[i, j]
                std_string = f"\n±({round(std, decimals)})" if std != float("inf") else ""  # type: ignore
                annot_matrix.loc[i, j] = f"{round(mean, decimals)}{std_string}"  # type: ignore
        annot = annot_matrix
        fmt = ""  # cannot format like before, as we do not only have a number
    else:
        fmt = f".{decimals}f"
        annot = True

    return sns.heatmap(pivot_table, ax=ax,
                       annot=annot, fmt=fmt,
                       annot_kws={'fontsize': config.plotstyle.matrix_font.size},
                       xticklabels=get_ticklabels(pivot_table.columns, usetex, xticks_log),
                       yticklabels=get_ticklabels(pivot_table.index, usetex, yticks_log),
                       cbar=False, vmin=vmin, vmax=vmax)


def get_all_num_rows_and_their_names(dataframe_list: list[pd.DataFrame], config):
    n_rows: list[int] = []
    row_names: list[list[str]] = []
    for i, df in enumerate(dataframe_list):
        x_axis = config.plot.x_axis[i]
        y_axis = config.plot.y_axis[0]
        metrics = df["evaluation.plot.metric"].unique()
        ignored_cols = [x_axis, y_axis]
        ignored_cols += list(metrics)
        ignored_cols += config.get("ignore_keys", [])
        ignored_cols += config.get("aggregate_groups", [])
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
    whitelisted_cols: list[str] | Literal["all"] = "all"  # everything is whitelisted if this value stays 'all'
    if isinstance(config.split_groups, list):
        whitelisted_cols = config.split_groups[:]
    elif config.split_groups is False:
        whitelisted_cols = []

    columns_with_non_unique_values = []
    for col in dataframe.columns:
        is_eval_key = col.startswith("evaluation.")
        is_ignored = col in ignored_cols
        is_whitelisted = whitelisted_cols == "all" or col in whitelisted_cols
        if any([is_ignored, is_eval_key, not is_whitelisted]):
            if is_whitelisted:
                log_warn(f"{col} is in the whitelist, but will be ignored. Probably {col} is in both 'split_groups' and 'aggregate_groups'.")
            log_debug(f"ignoring {col}")
            continue
        nunique = dataframe[col].nunique(dropna=False)
        if nunique > 1:
            log_debug(f"adding {col} since there are {nunique} unique values")
            for unique_hp in dataframe[col].unique():
                columns_with_non_unique_values.append(f"{col}={unique_hp}")
            necesarry_rows += (nunique)  # each unique parameter should be an individal plot

    rows_number = max(necesarry_rows, 1)
    col_names = columns_with_non_unique_values
    log_debug(f"{rows_number=}")
    log_debug(f"{col_names=}")

    return rows_number, col_names


def find_global_vmin_vmax(dataframe_list, config):
    vmin: int | float | None = None
    vmax: int | float | None = None
    num_cols = len(dataframe_list)

    if num_cols > 1:
        # all subplots should have same colors -> we need to find the limits
        vmin = float('inf')
        vmax = float('-inf')

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

            log_debug("colorbar_limits:\n" +
                            f"  subfigure number {i+1}, checking for metric {key}: \n" +
                            f"  min value is {min_value_present_in_current_df},\n" +
                            f"  max value is {max_value_present_in_current_df}")
            vmin = min(vmin, min_value_present_in_current_df)
            vmax = max(vmax, max_value_present_in_current_df)

    return vmin, vmax


def get_local_vmin_vmax(pt_object: PivotTable) -> tuple[float, float] | None:
    """replace vmin and vmax with custom values if specified"""
    dataframe = pt_object.grouped_df
    if "evaluation.plot.limits" in dataframe.columns:
        limits = dataframe["evaluation.plot.limits"].iloc[0]
        if limits:
            vmin = min(limits)
            vmax = max(limits)
            log_debug(f"setting cbar limits to {vmin}, {vmax} ")
            return vmin, vmax


def create_pivot_tables(
        dataframe_list: list[pd.DataFrame],
        config: AttributeDict,
        n_rows: list[int],
        row_names_by_col: list[list[str]]
    ) -> list[list[PivotTable]]:
    pivot_tables = [list() for _ in range(len(dataframe_list))]

    for i in range(len(dataframe_list)):
        row_names = row_names_by_col[i]
        for j in range(n_rows[i]):
            dataframe = dataframe_list[i]
            df_entry = dataframe.iloc[0]
            metric_name = df_entry["evaluation.plot.metric"]
            cols = config.plot.x_axis[i]
            idx = config.plot.y_axis[j]

            # df grouping for split_groups
            row_name = row_names[j]
            if row_name != "default":
                pname, pvalue = row_name.split("=", maxsplit=1)
                if pd.api.types.is_numeric_dtype(dataframe[pname]):
                    try:
                        pvalue = float(pvalue)
                    except ValueError:
                        pass
                try:
                    grouped_df = dataframe.groupby([pname]).get_group((pvalue,))
                except KeyError:
                    log_warn(f"Was not able to groupby '{pname}'," +
                              "maybe the data was created with different versions of FOB; skipping this row")
                    log_debug(f"{pname=}{pvalue=}{dataframe.columns=}{dataframe[pname]=}")
                    pivot_tables[i].append(PivotTable(pd.DataFrame(), pd.DataFrame()))
                    continue
            else:
                grouped_df = dataframe

            pivot_table = pd.pivot_table(grouped_df,
                                         columns=cols, index=idx,
                                         values=metric_name,
                                         aggfunc="mean")

            if config.plot.std:
                pivot_table_agg = pd.pivot_table(grouped_df,
                                                 columns=cols, index=idx,
                                                 values=metric_name,
                                                 aggfunc=config.plot.aggfunc,  fill_value=float("inf"), dropna=False)
            else:
                pivot_table_agg = None

            pivot_tables[i].append(PivotTable(grouped_df, pivot_table, pivot_table_agg))

    return pivot_tables


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
    if config.split_groups is False:
        n_rows_max = 1
        row_names = [["default"] for _ in range(num_cols)]
    else:
        n_rows_max = max(n_rows)

    log_debug(f"{n_rows=} and {num_cols=}")

    # TODO adjust figsize based on pivot table dimensions
    scale = config.plotstyle.scale
    if num_cols == 1 and n_rows_max > 1:
        figsize = (2**3 * scale, 2 * 3 * n_rows_max * scale)
    elif num_cols == 2:
        figsize = (12 * scale, 5.4 * n_rows_max * scale)
    elif num_cols > 2:
        figsize = (12 * (num_cols / 2) * scale, 5.4 * n_rows_max * scale)
    else:
        figsize = None

    # create pivot tables and extract subplot widths
    pivot_tables = create_pivot_tables(dataframe_list, config, n_rows, row_names)
    widths = []
    for i in range(num_cols):
        for j in range(n_rows[i]):
            widths.append(len(pivot_tables[i][j].pivot_table.columns))
    total_width = sum(widths)
    width_ratios = [0.975 * w/total_width for w in widths]
    # TODO: find better way to set width, so it is fixed for different sizes
    width_ratios.append(0.025)  # For colorbar

    # TODO: use seaborn FacetGrid
    fig, axs = plt.subplots(n_rows_max, num_cols + 1, figsize=figsize, width_ratios=width_ratios)
    if n_rows_max == 1:
        axs = [axs]

    # Adjust left and right margins as needed
    # fig.subplots_adjust(left=0.1, right=0.9, top=0.97, hspace=0.38, bottom=0.05,wspace=0.3)

    # None -> plt will chose vmin and vmax
    global_vmin, global_vmax = find_global_vmin_vmax(dataframe_list, config)

    for i in range(num_cols):
        num_rows: int = n_rows[i] if config.split_groups else 1
        for j in range(num_rows):
            pt_object = pivot_tables[i][j]
            ax = axs[j][i]

            # use correct formatting
            vmin, vmax = handle_formatting(pt_object, global_vmin, global_vmax)

            # up to here limits was the min and max over all dataframes, usually we want to use user values
            local_limits = get_local_vmin_vmax(pt_object)
            if local_limits is not None:
                vmin, vmax = local_limits

            create_one_grid_element(pt_object,
                                    config,
                                    ax, i, j,
                                    max_i=num_cols,
                                    max_j=num_rows,
                                    vmin=vmin,
                                    vmax=vmax,
                                    row_names=row_names)
    # add colorbars
    for j in range(num_rows):
        pt_object = pivot_tables[num_cols - 1][j]
        ax = axs[j][num_cols]
        df = pt_object.grouped_df
        vmin, vmax = handle_formatting(pt_object, global_vmin, global_vmax)
        local_limits = get_local_vmin_vmax(pt_object)
        if local_limits is not None:
            vmin, vmax = local_limits
        make_colorbar(df, config, ax, vmin, vmax)

    if config.plotstyle.tight_layout:
        fig.tight_layout()
    # SUPTITLE (the super title on top of the whole figure in the middle)
    # # TODO super title might be squished when used together with tight layout (removing for now)
    # if n_rows_max > 1 or num_cols > 1:
    #     # set experiment name as title when multiple matrices in image
    #     if config.experiment_name:
    #         fig.suptitle(config.experiment_name)
    return fig, axs


def create_one_grid_element(
        pt_object: PivotTable,
        config: AttributeDict,
        ax, i: int, j: int, max_i: int, max_j: int, vmin, vmax, row_names
    ):
    """does one 'axs' element as it is called in plt"""
    if pt_object.empty:
        return False
    current_dataframe = pt_object.grouped_df

    cols = config.plot.x_axis[i]
    idx = config.plot.y_axis[0]

    model_param = row_names[i][j]
    
    # optionally convert axis-tick-labels to logscale
    xticks_log = config.plotstyle.x_axis_labels_log10[i]
    yticks_log = config.plotstyle.y_axis_labels_log10[j]
    current_plot = create_matrix_plot(pt_object, config,
                                      cols, idx, ax=ax,
                                      vmin=vmin, vmax=vmax,
                                      xticks_log=xticks_log, yticks_log=yticks_log)

    # LABELS
    # Pretty name for label "learning_rate" => "Learning Rate"
    # remove x_label of all but last row, remove y_label for all but first column    
    if i > 0:
        current_plot.set_ylabel('', labelpad=8)
    else:
        current_plot.set_ylabel(pretty_name(current_plot.get_ylabel()))
    if j < max_j - 1:
        current_plot.set_xlabel('', labelpad=8)
    else:
        current_plot.set_xlabel(pretty_name(current_plot.get_xlabel()))

    # TITLE
    # title (heading) of the heatmap: <optimname> on <taskname> (+ additional info)
    if config.column_titles is None:
        # reading optimizer and task name after grouping
        df_entry = current_dataframe.iloc[0]  # just get an arbitrary trial
        opti_name = df_entry['optimizer.name']
        task_name = df_entry['task.name']
        title = f"{pretty_name(opti_name)} on {pretty_name(task_name)}"
    else:
        title = config.column_titles[i]
    if (max_i > 1 or max_j > 1) and model_param != "default":
        title += f"\n{model_param}"
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
    task_name = "_".join(sorted(set(task_names)))
    optim_name = "_".join(sorted(set(optim_names)))

    here = Path(__file__).parent.resolve()

    output_dir = Path(config.output_dir) if config.output_dir else here
    experiment_name = Path(config.experiment_name) if config.experiment_name else f"{optim_name}-{task_name}"
    output_file_path = output_dir / experiment_name

    return Path(f"{output_file_path}-{suffix}" if suffix else output_file_path)


def set_plotstyle(config: AttributeDict):
    plt.rcParams["text.usetex"] = config.plotstyle.text.usetex
    plt.rcParams["font.family"] = config.plotstyle.font.family
    plt.rcParams["font.size"] = config.plotstyle.font.size

def pretty_name(name: str, pretty_names: dict | str = {}) -> str:  # type: ignore pylint: disable=dangerous-default-value
    """
    Tries to use a mapping for the name, else will do some general replacement.
    mapping can be a directory or a filename of a yaml file with 'names' key
    """

    # reading from yaml and caching the dictionary
    label_file: Path = evaluation_path() / "labels.yaml"
    if isinstance(pretty_names, str):
        label_file = Path(pretty_names)

    if pretty_names == {} or isinstance(pretty_names, str):
        yaml_parser = YAMLParser()
        yaml_content = yaml_parser.parse_yaml(label_file)
        pretty_names: dict[str, str] = yaml_content["names"]

    # applying pretty names
    name_without_yaml_prefix = name.split(".")[-1]
    if name in pretty_names.keys():
        name = pretty_names[name]
    elif name_without_yaml_prefix in pretty_names.keys():
        name = pretty_names[name_without_yaml_prefix]
    else:
        name = name.replace('_', ' ').title()
    return name


def save_csv(dfs: list[pd.DataFrame], output_filename: Path):
    for i, df in enumerate(dfs):
        csv_output_filename = f"{output_filename.resolve()}-{i}.csv"
        log_info(f"saving raw data as {csv_output_filename}")
        df.to_csv(path_or_buf=csv_output_filename, index=False)


def save_plot(fig: Figure, output_file_path: Path, file_type: str, dpi: int):
    plot_output_filename = f"{output_file_path.resolve()}.{file_type}"
    log_info(f"saving figure as <{plot_output_filename}>")
    fig.savefig(plot_output_filename, dpi=dpi)


def save_files(fig, dfs: list[pd.DataFrame], output_file_path: Path, config: AttributeDict):
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    for file_type in config.output_types:
        if file_type == "csv":
            save_csv(dfs, output_file_path)
        elif file_type == "png" or file_type == "pdf":
            save_plot(fig, output_file_path, file_type, config.plotstyle.dpi)


def clean_config(config: AttributeDict) -> AttributeDict:
    """some processing that allows the user to be lazy, shortcut for the namespace, hidden values are found and config.all_values"""
    if "evaluation" in config.keys():
        evaluation_config: AttributeDict = config.evaluation
        evaluation_config["all_values"] = config
        config = evaluation_config
    else:
        log_warn("there is no 'evaluation' in the yaml provided!")
    if "data_dirs" in config.keys():
        value_is_none = not config.data_dirs
        value_has_wrong_type = not isinstance(config.data_dirs, (PathLike, str, list))
        if value_is_none or value_has_wrong_type:
            raise ValueError(f"Error: 'evaluation.data_dirs' was not provided correctly! check for typos in the yaml provided! value given: {config.data_dirs}")

    # allow the user to write a single string instead of a list of strings
    if not isinstance(config.output_types, list):
        config["output_types"] = [config.output_types]
        log_info("fixing value for key <config.output_types> to be a list[str]")

    if not isinstance(config.data_dirs, list):
        config["data_dirs"] = [Path(config.data_dirs)]
        log_info("fixing value for key <config.data_dirs> to be a list[Path]")

    # x_axis
    if not isinstance(config.plot.x_axis, list):
        config["plot"]["x_axis"] = [config.plot.x_axis]
        log_info("fixing value for key <config.plot.x_axis> to be a list[str]")
    if len(config.plot.x_axis) < len(config.data_dirs):
        # use same x axis for all if only one given
        missing_elements = len(config.data_dirs) - len(config.plot.x_axis)
        config["plot"]["x_axis"] += repeat(config.plot.x_axis[0], missing_elements)

    # y_axis
    if not isinstance(config.plot.y_axis, list):
        config["plot"]["y_axis"] = [config.plot.y_axis]
        log_info("fixing value for key <config.plot.y_axis> to be a list[str]")
    if len(config.plot.y_axis) < len(config.data_dirs):
        # use same x axis for all if only one given
        missing_elements = len(config.data_dirs) - len(config.plot.y_axis)
        config["plot"]["y_axis"] += repeat(config.plot.y_axis[0], missing_elements)

    return config


def main(config: AttributeDict):
    config = clean_config(config)  # sets config to config.evaluation, cleans some data
    workloads: List[Path] = [Path(name) for name in config.data_dirs]
    log_debug(f"{workloads}=")

    set_plotstyle(config)

    dfs = extract_dataframes(workloads, depth=config.depth, config=config)
    fig, _ = create_figure(dfs, config)

    output_file_path = get_output_file_path(dfs, config)

    save_files(fig, dfs, output_file_path, config)
