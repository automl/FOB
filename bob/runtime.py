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
        self.cpu_cores: int = args.cpu_cores if args.cpu_cores else cpu_count()


class RuntimeArgs(DatasetArgs):
    """
    Hold runtime specific arguments which is globally available information
    """
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.hyperparameter_path: Optional[Path] = args.hyperparameters
        self.output_dir: Optional[Path] = args.output
        self.checkpoint_dir: Optional[Path] = args.checkpoints
        self.submission_name: str = args.submission
