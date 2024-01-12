import argparse
import random
from typing import Optional
from pathlib import Path
from multiprocessing import cpu_count
from itertools import count

class DatasetArgs:
    """
    Hold arguments required for dataset creation
    """
    def __init__(self, args: argparse.Namespace):
        self.data_dir: Path = args.data_dir
        self.workload_name: str = args.workload
        self.workers: int = args.workers if args.workers else cpu_count() - 1


class RuntimeArgs(DatasetArgs):
    """
    Hold runtime specific arguments which is globally available information
    """
    _id = count(0)
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.submission_name: str = args.submission
        self.hyperparameter_path: Optional[Path] = args.hyperparameters
        self.resume: Optional[Path] = args.resume
        self.seed: int
        self.trial: int
        self.trial = args.start_trial + next(self._id)
        if args.seed_mode == "fixed":
            self.seed = args.seed
        elif args.seed_mode == "increment":
            self.seed = args.seed + self.trial - 1
        elif args.seed_mode == "random":
            self.seed = random.randint(0, 2**31)
        else:
            raise ValueError(f"unknown option for seed_mode, got: {args.seed_mode}.")
        output_dir = args.output if args.output else Path.cwd() / "experiments"
        self.output_dir: Path  = output_dir / self.submission_name / self.workload_name / f"trial_{self.trial}"
        self.checkpoint_dir: Path = self.output_dir / "checkpoints"
