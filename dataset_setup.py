import argparse
from pathlib import Path
from engine.engine import Engine, Run


def download_single_data(run: Run):
    data_module = run.get_datamodule()
    data_module.prepare_data()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_file", type=Path,
                        help="The yaml file specifying the experiment.")
    return parser


def main(args: argparse.Namespace, extra_args: list[str]):
    engine = Engine()
    engine.parse_experiment(args.experiment_file, extra_args=extra_args)
    done = set()
    for run in engine.runs():
        if run.workload.name in done:
            continue
        print(f"Setting up data for {run.workload_key} '{run.workload.name}'.")
        download_single_data(run)
        done.add(run.workload.name)


if __name__ == '__main__':
    parser = get_parser()
    args, extra_args = parser.parse_known_args()
    main(args, extra_args)
