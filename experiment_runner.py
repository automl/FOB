from pathlib import Path
import argparse
from lightning_utilities.core.rank_zero import rank_zero_info

from engine.engine import Engine
from engine.utils import begin_timeout


def main(args: argparse.Namespace, extra_args: list[str]):
    engine = Engine()
    engine.parse_experiment(args.experiment_file, extra_args=extra_args)
    engine.run_experiment()

    if args.send_timeout:
        rank_zero_info("submission_runner.py finished! Setting timeout of 10 seconds, as tqdm sometimes is stuck\n")
        begin_timeout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="runs an experiment specified by a file"
    )
    parser.add_argument("experiment_file", type=Path,
                        help="The yaml file specifying the experiment.")
    parser.add_argument("--send_timeout", action="store_true",
                        help="send a timeout after finishing this script (if you have problems with tqdm being stuck)")
    args, extra_args = parser.parse_known_args()
    main(args, extra_args)
