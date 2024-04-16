from pathlib import Path
import argparse
from engine.engine import Engine
from engine.parser import YAMLParser
from engine.utils import AttributeDict, convert_type_inside_dict
from evaluation import evaluation_path
from evaluation.plot import main

if __name__ == "__main__":
    # default path to settings
    default_yaml = evaluation_path() / "default.yaml"

    # parsing
    parser = argparse.ArgumentParser(description="Create a heatmap plot of benchmarking results.")
    parser.add_argument("settings", type=Path,
                        help="Path to the experiment yaml file.")
    parser.add_argument("--backend", type=str, choices=["engine", "legacy"], default="engine")
    args, extra_args = parser.parse_known_args()
    if args.backend == "engine":
        engine = Engine()
        engine.parse_experiment_from_file(args.settings, extra_args=extra_args)
        engine.plot()
    else:
        yaml_parser = YAMLParser()
        config = yaml_parser.parse_yamls_and_extra_args(default_yaml, args.settings, extra_args)
        config = convert_type_inside_dict(config, src=dict, tgt=AttributeDict)
        config = AttributeDict(config)

        main(config)
