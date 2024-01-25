# generates hyperparameter based on a grid search-space

import argparse
import json
from pathlib import Path
import sys
import runtime.grid_search


def main(args):
    search_space: Path = args.search_space
    output_dir: Path = args.output_dir
    if not search_space.is_file():
        print("Search space is not a file!", file=sys.stderr)
        sys.exit(27)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime.grid_search.write_hyperparameter(search_space, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="generates hyperparameter based on a grid search-space"
    )
    parser.add_argument("--search_space", "-s", type=Path, required=True,
                        help="path to a search space file")
    parser.add_argument("--output_dir", "-o", type=Path, required=True,
                        help="path where the hyperparameter files are stored")
    args = parser.parse_args()
    main(args)
