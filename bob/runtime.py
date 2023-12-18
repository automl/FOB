import argparse
from typing import Optional, Any
from pathlib import Path
from multiprocessing import cpu_count

class RuntimeArgs:
    """
    Hold runtime specific arguments which is globally available information
    """
    def __init__(self, args: argparse.Namespace) -> None:
        self.dataset_dir: Path = args.datasets
        self.hyperparameter_path: Optional[Path] = args.hyperparameters
        self.output_dir: Optional[Path] = args.output
        self.checkpoint_dir: Optional[Path] = args.checkpoints
        self.workload_name: str = args.workload
        self.submission_name: str = args.submission
        self.cpu_cores: int = args.cpu_cores if args.cpu_cores else cpu_count()
