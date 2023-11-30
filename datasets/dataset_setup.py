"""
Example command:

python3 datasets/dataset_setup.py \
    --data_dir=~/data_bob \
    --all
"""
import os
import argparse
from torchvision.datasets import CIFAR10, CIFAR100, MNIST


def download_mnist(data_dir):
    # https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
    # training set
    MNIST(root=data_dir, train=True, download=True)
    # test set
    MNIST(root=data_dir, train=False, download=True)


def download_cifar10(data_dir):
    # https://pytorch.org/vision/0.8/datasets.html#cifar
    # training set
    CIFAR10(root=data_dir, train=True, download=True)
    # test set
    CIFAR10(root=data_dir, train=False, download=True)


def download_cifar100(data_dir):
    # https://pytorch.org/vision/0.8/datasets.html#cifar
    # training set
    CIFAR100(root=data_dir, train=True, download=True)
    # test set
    CIFAR100(root=data_dir, train=False, download=True)


def get_parser():
    parser = argparse.ArgumentParser()

    # dir paths
    parser.add_argument(
        "--data_dir",
        help="The path to the folder where datasets should be stored.",
        required=True,
        type=str
    )
    parser.add_argument(
        "--tmp_dir",
        help="A local path to a folder where tmp files can be downloaded.",
        type=str,
    )

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
        "--cifar10",
        help="Whether to include CIFAR10",
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
    if args.all or args.cifar10:
        download_cifar10(args.data_dir)
    if args.all or args.cifar100:
        download_cifar100(args.data_dir)


if __name__ == '__main__':
    main()
