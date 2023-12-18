"""
Example command:

python3 datasets/dataset_setup.py \
    --data_dir=~/data_bob \
    --all
"""
import os
import argparse
from pathlib import Path
from workloads.mnist.data import MNISTDataModule
from workloads.coco.data import COCODataModule
from workloads.cifar100.data import CIFAR100DataModule


def download_mnist(data_dir):
    MNISTDataModule(data_dir).prepare_data()

def download_cifar100(data_dir):
    CIFAR100DataModule(data_dir).prepare_data()

def download_coco(data_dir):
    COCODataModule(data_dir).prepare_data()

def get_parser():
    parser = argparse.ArgumentParser()

    # dir paths
    parser.add_argument(
        "--data_dir", "-d",
        help="The path to the folder where datasets should be stored.",
        required=True,
        type=Path
    )
    parser.add_argument(
        "--tmp_dir",
        help="A local path to a folder where tmp files can be downloaded.",
        type=str,
    ) # TODO: actually use this

    # Workload args
    parser.add_argument(
        "--all",
        help="Whether to include all data sets",
        action="store_true"
    )
    parser.add_argument(
        "--mnist",
        help="Whether to include MNIST",
        action="store_true"
    )
    parser.add_argument(
        "--coco",
        help="Whether to include COCO",
        action="store_true"
    )
    parser.add_argument(
        "--cifar100",
        help="Whether to include CIFAR100",
        action="store_true"
    )
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # TODO: check if data_dir already exists

    # if no tmp folder given: create tmp folder in data folder
    if not args.tmp_dir:
        DEFAULT_TMP_DIR = "tmp/"
        args.tmp_dir = os.path.join(args.data_dir, DEFAULT_TMP_DIR)
        # TODO make sure tmp_dir exists

    if args.all or args.mnist:
        download_mnist(args.data_dir)
    if args.all or args.coco:
        download_coco(args.data_dir)
    if args.all or args.cifar100:
        download_cifar100(args.data_dir)


if __name__ == '__main__':
    main()
