import json
from pathlib import Path
import argparse

from pytorch_fob.engine.engine import Engine

def get_config(intensifier_file: Path, runhistory_file: Path) -> dict:
    with open(intensifier_file, "r", encoding="utf8") as f:
        content = json.load(f)
        # get the best id and the trajectory
        incumbent_ids = content["incumbent_ids"]
        intensifier_id = incumbent_ids[0]
        trajectory = content["trajectory"]

        # take the last one of the trajectory and make sure its our intensifier
        incumbent_info = trajectory[-1]
        assert incumbent_info["config_ids"][0] == intensifier_id

    with open(runhistory_file, "r", encoding="utf8") as f:
        content = json.load(f)
        configs = content["configs"]
        config = configs[str(intensifier_id)]

    return config


def config_to_fob_input(config: dict):
    arglist = [f"{k}={v}" for k, v in config.items()]
    # maybe add something like this as arg?
    arglist += [
       "engine.test=true",
       "engine.plot=false",
       "engine.resume=true"
    ]
    return arglist


def main(args: argparse.Namespace, extra_args: list[str]):
    intensifier_file = args.smac_directory / "intensifier.json"
    runhistory_file = args.smac_directory / "runhistory.json"

    config = get_config(intensifier_file, runhistory_file)
    print(config)

    print()

    fob_input = config_to_fob_input(config)
    print(fob_input)

    engine = Engine()
    engine.parse_experiment_from_file(args.experiment_file, extra_args=extra_args + fob_input)
    engine.run_experiment()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="gets the config of the best incumbent"
    )
    parser.add_argument("experiment_file", type=Path,
                        help="The yaml file specifying the experiment.")
    parser.add_argument("smac_directory", type=Path,
                        help="The directory that hold the intensifier.json and runhistory.json")
    args, extra_args = parser.parse_known_args()
    main(args, extra_args)
