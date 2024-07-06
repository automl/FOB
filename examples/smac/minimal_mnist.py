import math
from pathlib import Path
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


def config_space() -> ConfigurationSpace:
    cs = ConfigurationSpace()
    cs.add_hyperparameter(Categorical("optimizer.name", ["adamw_baseline", "sgd_baseline"]))
    cs.add_hyperparameter(Float("optimizer.learning_rate", (1.e-5, 1.e-1), log=True))
    cs.add_hyperparameter(Float("optimizer.weight_decay", (1.e-5, 1.e-1), log=True))
    cs.add_hyperparameter(Float("optimizer.momentum", (0, 1)))
    cs.add_hyperparameter(Float("optimizer.one_minus_beta1", (1e-3, 1e-1)))
    cs.add_hyperparameter(Float("optimizer.beta2", (0, 1)))
    cs.add_hyperparameter(Float("optimizer.eta_min_factor", (1.e-5, 1.e-1), log=True))
    cs.add_hyperparameter(Float("optimizer.warmup_factor", (1.e-4, 1.e-0), log=True))
    cs.add_hyperparameter(Integer("task.model.num_hidden", (32, 512), log=True))
    cs.add_hyperparameter(Categorical("task.model.activation", ["relu", "sigmoid"]))

    # conditions
    cs.add_condition(EqualsCondition(cs["optimizer.momentum"], cs["optimizer.name"], "sgd_baseline"))
    cs.add_condition(EqualsCondition(cs["optimizer.one_minus_beta1"], cs["optimizer.name"], "adamw_baseline"))
    cs.add_condition(EqualsCondition(cs["optimizer.beta2"], cs["optimizer.name"], "adamw_baseline"))
    cs.add_condition(EqualsCondition(cs["optimizer.warmup_factor"], cs["optimizer.name"], "adamw_baseline"))
    return cs

def train_mnist(config: Configuration, seed: int, budget: float) -> float:
    config_dict = config.get_dictionary()
    if "optimizer.one_minus_beta1" in config_dict:
        config_dict["optimizer.beta1"] = 1 - config_dict["optimizer.one_minus_beta1"]
    arglist = [f"{k}={v}" for k, v in config_dict.items()]
    arglist += [
        f"engine.restrict_train_epochs={math.ceil(budget)}",
        "engine.test=false",
        "engine.validate=true",
        "engine.output_dir=./examples/smac/outputs/fob",
        "engine.data_dir=./examples/smac/data",
        f"engine.seed={seed}",
        "task.name=mnist",
    ]
    engine = Engine()
    engine.parse_experiment({}, extra_args=arglist)
    run = next(engine.runs())  # only get one run
    score = run.start()
    return 1 - sum(map(lambda x: x["val_acc"], score["validation"])) / len(score["validation"])


def run_smac():
    configspace = config_space()
    scenario = Scenario(
        name="FOB_HPO",
        configspace=configspace,
        deterministic=True,
        output_directory=Path("examples", "smac", "outputs", "smac"),
        seed=42,
        n_trials=100,
        max_budget=30,
        min_budget=1,
        n_workers=1,
        walltime_limit=3600
    )
    smac = SMAC4MF(
        target_function=train_mnist,
        scenario=scenario,
        initial_design=SMAC4MF.get_initial_design(scenario=scenario, n_configs=2),
        intensifier=Hyperband(
            scenario=scenario,
            incumbent_selection="highest_budget",
            eta=3,
        ),
        overwrite=True
    )
    incumbent = smac.optimize()
    return incumbent


if __name__ == "__main__":
    engine = Engine()
    engine.parse_experiment({"task": {"name": "mnist"}, "engine": {"data_dir": "./examples/smac/data"}})
    engine.prepare_data()
    incumbent = run_smac()
    print(incumbent)
