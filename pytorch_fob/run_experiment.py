from pathlib import Path
import argparse
import logging

from pytorch_fob.engine.engine import Engine


def main(args: argparse.Namespace, extra_args: list[str]):
    engine = Engine()
    engine.parse_experiment_from_file(args.experiment_file, extra_args=extra_args)
    engine.run_experiment()
    engine.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="runs an experiment specified by a file"
    )
    parser.add_argument("experiment_file", type=Path,
                        help="The yaml file specifying the experiment.")
    parser.add_argument("--log_level", type=str, choices=["debug", "info", "warn", "silent"], default="info",
                        help="Set the log level")
    args, extra_args = parser.parse_known_args()
    pytorch_logger = logging.getLogger("lightning.pytorch")
    match args.log_level:
        case "debug":
            pytorch_logger.setLevel(logging.DEBUG)
        case "info":
            pytorch_logger.setLevel(logging.INFO)
        case "warn":
            pytorch_logger.setLevel(logging.WARNING)
        case "silent":
            pytorch_logger.setLevel(logging.CRITICAL)
    main(args, extra_args)
