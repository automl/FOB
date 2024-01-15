"""
Example command:

python3 datasets/dataset_setup.py \
    --data_dir=~/data_bob \
    --all
"""
import os
import argparse
from pathlib import Path
from runtime import DatasetArgs
import workloads

def download_single_data(args: argparse.Namespace):
    dataset_args = DatasetArgs(args)
    workload = workloads.import_workload(dataset_args.workload_name)
    data_module: workloads.WorkloadDataModule = workload.get_datamodule(dataset_args)
    data_module.prepare_data()

def download_all_data(args: argparse.Namespace):
    for workload_name in workloads.workload_names():
        args.workload = workload_name
        download_single_data(args)

def get_parser():
    parser = argparse.ArgumentParser()

    # dir paths
    parser.add_argument("--data_dir", "-d", required=True, type=Path, \
                        help="path to folder where datasets should be downloaded to (should be workload independent).")
    parser.add_argument("--workload", "-w", type=str, choices=workloads.workload_names(), \
                        help="the workload for which to download the data.")
    parser.add_argument("--all", "-a", action="store_true", \
                        help="Whether to download all datasets.")
    parser.add_argument("--workers", type=int, \
                        help="number of parallelism used for loading data, default: all available")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.all:
        download_all_data(args)
    elif args.workload is not None:
        download_single_data(args)
    else:
        raise ValueError("Must specify which workloads to download.")


if __name__ == '__main__':
    main()
