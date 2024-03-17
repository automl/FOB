from pathlib import Path
import argparse
from engine.parser import YAMLParser
from engine.utils import AttributeDict, convert_type_inside_dict
from evaluation.plot import main

if __name__ == "__main__":
    # default path to settings
    here = Path(__file__).parent.resolve()
    default_yaml = here / "evaluation" / "default.yaml"

    # parsing
    parser = argparse.ArgumentParser(description="Create a heatmap plot of benchmarking results.")
    parser.add_argument("settings", type=Path,
                        help="Path to the yaml that has the path to the data dir and additional plotting settings)")
    args, extra_args = parser.parse_known_args()
    yaml_parser = YAMLParser()
    config = yaml_parser.parse_yamls_and_extra_args(default_yaml, args.settings, extra_args)
    config = convert_type_inside_dict(config, src=dict, tgt=AttributeDict)
    config = AttributeDict(config)

    main(config)
