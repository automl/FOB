import argparse
import json
import random
from typing import Any, Optional
from pathlib import Path
from multiprocessing import cpu_count
from itertools import count
from runtime.utils import some
from runtime.grid_search import load_json, is_search_space, generate_hyperparameter_from_search_space
from submissions import submission_path


class DatasetArgs:
    """
    Hold arguments required for dataset creation
    """
    def __init__(self, args: argparse.Namespace):
        self.data_dir: Path = args.data_dir
        self.workload_name: str = args.workload
        self.workers: int = min(16, some(args.workers, default=cpu_count() - 1))

    def _resolve_hyperparameters(self, hyperparameters: str | Path | None, start_hyperparameter: int) -> bool:
        default_hparam_path = submission_path(self.submission_name) / "hyperparameters.json"
        hparams = some(hyperparameters, default=default_hparam_path)
        from_search_space = False
        if isinstance(hparams, Path) and hparams.is_dir():
            from_search_space = True
            try:
                hparams = list(hparams.iterdir())[start_hyperparameter + self.run_index]
            except IndexError as ie:
                print("Not enough hyperparameter files in specified folder.")
                raise ie
        hparams = load_json(hparams)
        if is_search_space(hparams):
            from_search_space = True
            try:
                hparams = list(generate_hyperparameter_from_search_space(hparams))
                hparams = hparams[start_hyperparameter + self.run_index]
            except IndexError as ie:
                print("Not enough hyperparameters in search space.")
                raise ie
        self.hyperparameters = hparams
        return from_search_space

    def export_settings(self):
        # write hyperparameters to outdir
        hparams_out = self.output_dir / "hyperparameters.json"
        with open(hparams_out, "w", encoding="utf8") as f:
            json.dump(self.hyperparameters, f, indent=4)
        # write runtime args
        out_dict = {k: str(v) if isinstance(v, Path) else v for k, v in self.__dict__.items()}
        with open(self.output_dir / "runtime_args.json", "w", encoding="utf8") as f:
            json.dump(out_dict, f, indent=4)


def hyperparameters(arg: str) -> Path | str:
    try:
        json.loads(arg)
        return arg
    except json.decoder.JSONDecodeError:
        return Path(arg)
