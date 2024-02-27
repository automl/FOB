from pathlib import Path
from typing import Any, Callable
import yaml
from submissions import submission_path
from workloads import workload_path
from .grid_search import gridsearch


def runtime_path() -> Path:
    return Path(__file__).resolve().parent


class Runtime():
    def __init__(self) -> None:
        self.experiments = []
        self.workload_key = "workload"
        self.submission_key = "submission"
        self.identifier_key = "name"
        self.default_file_name = "default.yaml"

    def parse_config(self, file: Path):
        searchspace = self._parse_yaml(file)
        self.experiments += gridsearch(searchspace)

    def fill_experiments_from_default(self):
        for i, _ in enumerate(self.experiments):
            # order from higher to lower in hierarchy
            self.experiments[i] = self._fill_unnamed_from_default(self.experiments[i], runtime_path)
            self.experiments[i] = self._fill_named_from_default(self.experiments[i], self.workload_key, workload_path)
            self.experiments[i] = self._fill_named_from_default(self.experiments[i], self.submission_key, submission_path)

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
        for i, e in enumerate(self.experiments):
            outpath = Path(e["runtime"]["output_dir"]) / f"experiment_{i}.yaml"
            outpath.parent.mkdir(parents=True, exist_ok=True)
            with open(outpath, "w", encoding="utf8") as f:
                yaml.safe_dump(e, f)
