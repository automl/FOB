import argparse
from pathlib import Path

import workloads
import submissions


def main(args: argparse.Namespace):
    dataset_path: Path = args.datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="runs a single submission (optimizer and scheduler) on a single workload")
    parser.add_argument("--datasets", "-d", required=True, type=Path, help="path to all datasets (should be workload independent)")
    parser.add_argument("--download", default=False, action="store_true", help="download dataset if it does not exist")
    parser.add_argument("--checkpoints", "-c", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--workload", "-w", required=True, type=str, choices=workloads.workload_names())
    parser.add_argument("--submission", "-s", required=True, type=str, choices=submissions.submission_names())
    # TODO: hyperparameter, trial number, experiment name
    args = parser.parse_args()
    main(args)
