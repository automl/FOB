import argparse
from typing import Optional
from pathlib import Path
from multiprocessing import cpu_count

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
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.submission_name: str = args.submission
        self.hyperparameter_path: Optional[Path] = args.hyperparameters
        output_dir = args.output if args.output else Path.cwd() / "experiments"
        self.output_dir: Path  = output_dir / self.submission_name / self.workload_name
        self.checkpoint_dir: Path = args.checkpoints if args.checkpoints else self.output_dir / "checkpoints"
