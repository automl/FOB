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


class RuntimeArgs(DatasetArgs):
    """
    Hold runtime specific arguments which is globally available information
    """
    _id = count(0)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.submission_name: str = args.submission
        self.resume: Optional[Path] = args.resume
        self.devices: Optional[int] = args.devices
        self.max_steps: Optional[int] = args.max_steps
        self.silent: bool = some(args.silent, default=False)
        self.run_index: int = next(self._id)
        self.trial: int = args.start_trial + self.run_index
        self.log_extra = args.log_extra
        output_dir = some(args.output, default=Path.cwd() / "experiments")
        self.output_dir: Path = output_dir / self.submission_name / self.workload_name / f"trial_{self.trial}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir: Path = self.output_dir / "checkpoints"
        self.test_only: bool = some(args.test_only, default=False)
        self.deterministic: bool = args.deterministic
        self.optimize_memory: bool = some(args.optimize_memory, default=False)
        self.hyperparameters: dict[str, Any]
        self.use_bfloat = not args.no_bfloat
        from_search_space = self._resolve_hyperparameters(args.hyperparameters, args.start_hyperparameter)
        if args.seed_mode == "fixed":
            self.seed = args.seed
        elif args.seed_mode == "increment":
            self.seed = args.seed + self.run_index
        elif args.seed_mode == "random":
            self.seed = random.randint(0, 2**31)
        else:
            if from_search_space:
                self.seed = args.seed
            else:
                self.seed = args.seed + self.run_index

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
