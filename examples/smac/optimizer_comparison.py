from pathlib import Path
import argparse
import logging
from smac.facade.multi_fidelity_facade import MultiFidelityFacade as SMAC4MF
from smac.intensifier.hyperband import Hyperband
from smac.scenario import Scenario
from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
    EqualsCondition,
    Categorical
)
from pytorch_fob import Engine

extra_args = None
experiment_file = None


def config_space() -> ConfigurationSpace:
    # TODO
    cs = ConfigurationSpace()
    cs.add_hyperparameter(Categorical("optimizer.name", ["adamw_baseline", "sgd_baseline"]))
    cs.add_hyperparameter(Float("optimizer.learning_rate", (1.e-5, 1.e-1), log=True))
    cs.add_hyperparameter(Float("optimizer.weight_decay", (1.e-5, 1.e-1), log=True))
    cs.add_hyperparameter(Float("optimizer.momentum", (0, 1)))
    cs.add_hyperparameter(Float("optimizer.one_minus_beta1", (1e-3, 1e-1)))
    cs.add_hyperparameter(Float("optimizer.beta2", (0, 1)))
    cs.add_hyperparameter(Float("optimizer.eta_min_factor", (1.e-5, 1.e-1), log=True))
    cs.add_hyperparameter(Float("optimizer.warmup_factor", (1.e-4, 1.e-0), log=True))

    # conditions
    cs.add_condition(EqualsCondition(cs["optimizer.momentum"], cs["optimizer.name"], "sgd_baseline"))
    cs.add_condition(EqualsCondition(cs["optimizer.one_minus_beta1"], cs["optimizer.name"], "adamw_baseline"))
    cs.add_condition(EqualsCondition(cs["optimizer.beta2"], cs["optimizer.name"], "adamw_baseline"))
    cs.add_condition(EqualsCondition(cs["optimizer.warmup_factor"], cs["optimizer.name"], "adamw_baseline"))
    return cs

# TODO: decorate
def train(config: Configuration, seed: int, budget: float) -> float:
    if extra_args is None or experiment_file is None:
        raise Exception("run as main file!")
    round_budget = round(budget)
    arglist = extra_args + [f"{k}={v}" for k, v in config.get_dictionary().items()]
    arglist += [
        f"engine.restrict_train_epochs={round_budget}",
        f"engine.seed={seed}",
    ]
    engine = Engine()
    engine.parse_experiment_from_file(experiment_file, extra_args=arglist)
    run = next(engine.runs())  # only get one run
    score = run.start()
    return 1 - sum(map(lambda x: x["val_acc"], score["validation"])) / len(score["validation"])


def run_smac(max_epochs: int):
    configspace = config_space()
    scenario = Scenario(
        name="FOB_HPO",
        configspace=configspace,
        deterministic=True,
        output_directory=Path("examples", "smac", "outputs", "optimizer_comparison"),
        seed=42,
        n_trials=250,
        max_budget=max_epochs,
        min_budget=1,
        n_workers=1, # TODO: https://github.com/automl/SMAC3/blob/main/examples/1_basics/7_parallelization_cluster.py
    )
    smac = SMAC4MF(
        target_function=train,
        scenario=scenario,
        initial_design=SMAC4MF.get_initial_design(scenario=scenario, n_configs=2),
        intensifier=Hyperband(
            scenario=scenario,
            incumbent_selection="highest_budget",
            eta=3,
        ),
        overwrite=True,
        dask_client=None, # TODO
    )
    incumbent = smac.optimize()
    return incumbent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="runs an experiment specified by a file"
    )
    parser.add_argument("experiment_file", type=Path,
                        help="The yaml file specifying the experiment.")
    parser.add_argument("--send_timeout", action="store_true",
                        help="send a timeout after finishing this script (if you have problems with tqdm being stuck)")
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
    experiment_file = args.experiment_file
    engine = Engine()
    engine.parse_experiment_from_file(experiment_file, extra_args=extra_args)
    engine.prepare_data()
    max_epochs = next(engine.runs()).task.max_epochs
    del engine
    incumbent = run_smac(max_epochs)
    print(incumbent)
