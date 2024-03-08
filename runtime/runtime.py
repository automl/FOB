from pathlib import Path
from typing import Any, Callable, Iterable
import hashlib
import re
import yaml
from submissions import Submission, submission_path, submission_names
from workloads import WorkloadModel, WorkloadDataModule, import_workload, workload_path, workload_names
from .grid_search import gridsearch
from .configs import RuntimeConfig, SubmissionConfig, WorkloadConfig
from .utils import path_to_str_inside_dict, dict_differences, concatenate_dict_keys


def runtime_path() -> Path:
    return Path(__file__).resolve().parent


class Run():
    def __init__(
            self,
            config: dict[str, Any],
            default_config: dict[str, Any],
            workload_key: str,
            submission_key: str,
            runtime_key: str,
            identifier_key: str
        ) -> None:
        self._config = config
        self.workload_key = workload_key
        self.submission_key = submission_key
        self.runtime_key = runtime_key
        self.runtime = RuntimeConfig(config, workload_key, runtime_key)
        self.submission = SubmissionConfig(config, submission_key, workload_key, identifier_key)
        self.workload = WorkloadConfig(config, workload_key, runtime_key, identifier_key)
        self._set_outpath(default_config)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _set_outpath(self, default_config: dict[str, Any]):
        base = self.runtime.output_dir / self.workload.output_dir_name / self.submission.output_dir_name
        exclude_keys = ["name", "output_dir_name"]
        include_runtime = ["deterministic", "optimize_memory", "seed"]
        exclude_keys += [k for k in self._config[self.runtime_key] if not k in include_runtime]
        print(exclude_keys)
        diffs = concatenate_dict_keys(dict_differences(self._config, default_config), exclude_keys=exclude_keys)
        print(diffs)
        run_dir = ",".join(f"{k}={str(v)}" for k, v in diffs.items()) if diffs else "default"
        if len(run_dir) > 254:  # max file name length
            hashdir = hashlib.md5(run_dir.encode()).hexdigest()
            print(f"Warning: folder name {run_dir} is too long, using {hashdir} instead.")
            run_dir = hashdir
        self.run_dir = base / run_dir

    def export_config(self):
        with open(self.run_dir / "config.yaml", "w", encoding="utf8") as f:
            yaml.safe_dump(path_to_str_inside_dict(self._config), f)

    def get_submission(self) -> Submission:
        return Submission(self.submission)

    def get_workload(self) -> tuple[WorkloadModel, WorkloadDataModule]:
        workload = import_workload(self.workload.name)
        return workload.get_workload(self.get_submission(), self.workload)

    def get_datamodule(self) -> WorkloadDataModule:
        workload = import_workload(self.workload.name)
        return workload.get_datamodule(self.workload)


class Runtime():
    def __init__(self) -> None:
        self._runs = []
        self._defaults = []
        self.workload_key = "workload"
        self.submission_key = "submission"
        self.runtime_key = "runtime"
        self.identifier_key = "name"
        self.default_file_name = "default.yaml"

    def parse_experiment(self, file: Path, extra_args: Iterable[str] = tuple()):
        searchspace = self._parse_yaml(file)
        for arg in extra_args:
            self._parse_into_searchspace(searchspace, arg)
        self._named_dicts_to_list(
            searchspace,
            [self.submission_key, self.workload_key],
            [submission_names(), workload_names()]
        )
        self._runs += gridsearch(searchspace)
        self._fill_runs_from_default(self._runs)
        self._fill_defaults()

    def runs(self) -> list[Run]:
        runs = []
        for config, default_config in zip(self._runs, self._defaults):
            run = Run(
                config,
                default_config,
                self.workload_key,
                self.submission_key,
                self.runtime_key,
                self.identifier_key
            )
            runs.append(run)
        return runs

    def _parse_into_searchspace(self, searchspace: dict[str, Any], arg: str):
        keys, value = arg.split("=")
        keys = keys.split(".")
        keys_with_list_indices = []
        for key in keys:
            match = re.search(r"^(.*?)\[(\-?\d+)\]$", key)
            if match:
                keys_with_list_indices.append(match.group(1))
                keys_with_list_indices.append(int(match.group(2)))
            else:
                keys_with_list_indices.append(key)
        target = searchspace
        for key in keys_with_list_indices[:-1]:
            target = target[key]
        target[keys_with_list_indices[-1]] = value

    def _named_dicts_to_list(self, searchspace: dict[str, Any], keys: list[str], valid_options: list[list[str]]):
        assert len(keys) == len(valid_options)
        for key, opts in zip(keys, valid_options):
            if isinstance(searchspace[key], dict) and all(name in opts for name in searchspace[key]):
                searchspace[key] = [cfg | {self.identifier_key: name} for name, cfg in searchspace[key].items()]

    def _fill_defaults(self):
        self._defaults = []
        for run in self._runs:
            default_cfg = {
                k: {self.identifier_key: run[k][self.identifier_key]}
                for k in [self.workload_key, self.submission_key]
            }
            self._defaults.append(default_cfg)
        self._fill_runs_from_default(self._defaults)

    def _fill_runs_from_default(self, runs: list[dict[str, Any]]):
        for i, _ in enumerate(runs):
            # order from higher to lower in hierarchy
            runs[i] = self._fill_unnamed_from_default(runs[i], runtime_path)
            runs[i] = self._fill_named_from_default(runs[i], self.workload_key, workload_path)
            runs[i] = self._fill_named_from_default(runs[i], self.submission_key, submission_path)

    def _fill_unnamed_from_default(self, experiment: dict[str, Any], unnamed_root: Callable) -> dict[str, Any]:
        default_path: Path = unnamed_root() / self.default_file_name
        default_config = self._parse_yaml(default_path)
        self._merge_dicts_hierarchical(default_config, experiment)
        return default_config

    def _fill_named_from_default(self, experiment: dict[str, Any], key: str, named_root: Callable) -> dict[str, Any]:
        self._argcheck_named(experiment, key, self.identifier_key)
        named = experiment[key]
        if isinstance(named, dict):
            named = named[self.identifier_key]
        else:
            experiment[key] = {self.identifier_key: named}
        default_path: Path = named_root(named) / self.default_file_name
        default_config = self._parse_yaml(default_path)
        self._merge_dicts_hierarchical(default_config, experiment)
        return default_config

    def _argcheck_named(self, experiment: dict[str, Any], key: str, identifier: str):
        assert key in experiment, f"You did not provide any {key}."
        assert isinstance(experiment[key], str) or identifier in experiment[key], \
            f"Unknown {key}, either specify only a string or provide a key '{identifier}'"

    def _merge_dicts_hierarchical(self, lo: dict, hi: dict):
        for k, v in hi.items():
            if isinstance(v, dict) and isinstance(lo.get(k, None), dict):
                self._merge_dicts_hierarchical(lo[k], v)
            else:
                lo[k] = v

    def _parse_yaml(self, file: Path):
        with open(file, "r", encoding="utf8") as f:
            return yaml.safe_load(f)

    def dump_experiments(self):
        # TODO: remove this and make proper export function
        for i, e in enumerate(self._runs):
            outpath = Path(e["runtime"]["output_dir"]) / f"experiment_{i}.yaml"
            outpath.parent.mkdir(parents=True, exist_ok=True)
            with open(outpath, "w", encoding="utf8") as f:
                yaml.safe_dump(e, f)
