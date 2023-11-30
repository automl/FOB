import argparse
import pathlib
import sys


def main(args: argparse.Namespace):
    data_path: pathlib.Path = args.datasets
    if not (data_path.exists() and data_path.is_dir()):
        print(f"Dataset path \"{data_path}\" does not exist!")
        sys.exit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", "-d", required=True, type=pathlib.Path, help="Path to all datasets (should be workload independent)")
    parser.add_argument("--checkpoints", "-c", type=pathlib.Path)
    parser.add_argument("--output", "-o", type=pathlib.Path)
    args = parser.parse_args()
    main(args)