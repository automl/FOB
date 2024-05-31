from pathlib import Path
import argparse
from smac.facade.multi_fidelity_facade import MultiFidelityFacade as SMAC4MF
from smac.intensifier.hyperband import Hyperband
from smac.scenario import Scenario
from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
    Constant
)
from pytorch_fob import Engine
from pytorch_fob.engine.utils import set_loglevel


def config_space(optimizer_name: str) -> ConfigurationSpace:
    cs = ConfigurationSpace()
    cs.add_hyperparameter(Constant("optimizer.name", optimizer_name))
    cs.add_hyperparameter(Float("optimizer.learning_rate", (1.e-5, 1.e-1), log=True))
    cs.add_hyperparameter(Float("optimizer.eta_min_factor", (1.e-3, 1.e-1), log=True))
    cs.add_hyperparameter(Float("optimizer.warmup_factor", (1.e-3, 1.e-0), log=True))
    if optimizer_name in ["adamw_baseline", "sgd_baseline"]:
        cs.add_hyperparameter(Float("optimizer.weight_decay", (1.e-5, 1.e-0), log=True))
    if optimizer_name in ["adamw_baseline", "adamcpr_fast"]:
        cs.add_hyperparameter(Float("optimizer.one_minus_beta1", (1e-2, 2e-1), log=True))
        cs.add_hyperparameter(Float("optimizer.beta2", (0.9, 0.999)))
    if optimizer_name == "sgd_baseline":
        cs.add_hyperparameter(Float("optimizer.momentum", (0, 1)))
    if optimizer_name == "adamcpr_fast":
        cs.add_hyperparameter(Integer("optimizer.kappa_init_param", (1, 19550), log=True))
        cs.add_hyperparameter(Constant("optimizer.kappa_init_method", "warm_start"))
    return cs


def get_target_fn(extra_args, experiment_file):
    def train(config: Configuration, seed: int, budget: float) -> float:
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
    return train


def run_smac(target_fn, optimizer_name: str, max_epochs: int, outdir: Path):
    configspace = config_space(optimizer_name)
    scenario = Scenario(
        name=f"FOB_HPO_{optimizer_name}",
        configspace=configspace,
        deterministic=True,
        output_directory=outdir / "smac",
        seed=42,
        n_trials=250,
        max_budget=max_epochs,
        min_budget=5,
        n_workers=1, # TODO: https://github.com/automl/SMAC3/blob/main/examples/1_basics/7_parallelization_cluster.py
    )
    smac = SMAC4MF(
        target_function=target_fn,
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
    parser.add_argument("--log_level", type=str, choices=["debug", "info", "warn", "silent"], default="info",
                        help="Set the log level")
    args, extra_args = parser.parse_known_args()
    set_loglevel(args.log_level)
    experiment_file = args.experiment_file
    engine = Engine()
    engine.parse_experiment_from_file(experiment_file, extra_args=extra_args)
    engine.prepare_data()
    run = next(engine.runs())
    max_epochs = run.task.max_epochs
    optimizer_name = run.optimizer.name
    outdir = run.engine.output_dir
    del engine
    incumbent = run_smac(get_target_fn(extra_args, experiment_file), optimizer_name, max_epochs, outdir)
    print(incumbent)
