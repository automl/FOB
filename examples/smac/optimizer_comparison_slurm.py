from pathlib import Path
import argparse
from argparse import Namespace
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
from dask.distributed import Client
from dask_jobqueue.slurm import SLURMCluster
from pytorch_fob import Engine
from pytorch_fob.engine.utils import set_loglevel, seconds_to_str, str_to_seconds


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


def run_smac(target_fn, args: Namespace, optimizer_name: str, max_epochs: int, outdir: Path,
             cores: int, max_time_per_job: str, devices: int, partition: str):
    configspace = config_space(optimizer_name)
    n_workers: int = args.n_workers
    scenario = Scenario(
        name=f"FOB_HPO_{optimizer_name}",
        configspace=configspace,
        deterministic=True,
        output_directory=outdir / "smac",
        seed=args.seed,
        n_trials=args.n_trials,
        max_budget=max_epochs,
        min_budget=args.min_budget,
        n_workers=n_workers, # TODO: https://github.com/automl/SMAC3/blob/main/examples/1_basics/7_parallelization_cluster.py
    )
    cluster = SLURMCluster(
        # More tips on this here: https://jobqueue.dask.org/en/latest/advanced-tips-and-tricks.html#how-to-handle-job-queueing-system-walltime-killing-workers
        # This is the partition of our slurm cluster.
        queue=partition,
        cores=cores,
        memory=f"{cores*2} GB",
        # Walltime limit for each worker. Ensure that your function evaluations
        # do not exceed this limit.
        walltime=sbatch_time(max_time_per_job, 1.1),
        job_extra_directives=[f"--gres=gpu:{devices}"],
        processes=1, # TODO: maybe number devices?
        log_directory=outdir / "smac" / "smac_dask_slurm",
        worker_extra_args=["--lifetime", str(str_to_seconds(max_time_per_job))],
    )
    cluster.adapt(minimum=1, maximum=n_workers, minimum_jobs=0, maximum_jobs=n_workers)
    print("waiting for cluster to schedule at least 1 job...")
    cluster.wait_for_workers(1, 60*10)
    print("cluster job script:", cluster.job_script())
    print("cluster logs:", cluster.get_logs())
    print("cluster status:", cluster.status)
    client = cluster.get_client()
    smac = SMAC4MF(
        target_function=target_fn,
        scenario=scenario,
        initial_design=SMAC4MF.get_initial_design(scenario=scenario, n_configs=2),
        intensifier=Hyperband(
            scenario=scenario,
            incumbent_selection="highest_budget",
            eta=args.eta,
        ),
        overwrite=True,
        dask_client=client,
    )
    incumbent = smac.optimize()
    return incumbent


def sbatch_time(time: str, time_factor: float) -> str:
    seconds = str_to_seconds(time) if isinstance(time, str) else time
    return seconds_to_str(int(time_factor * seconds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="runs an experiment specified by a file"
    )
    parser.add_argument("experiment_file", type=Path,
                        help="The yaml file specifying the experiment.")
    parser.add_argument("--log_level", type=str, choices=["debug", "info", "warn", "silent"], default="info",
                        help="Set the log level")
    parser.add_argument("--n_workers", type=int, default=4,
                        help="maximum number of parallel SMAC runs")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for SMAC")
    parser.add_argument("--n_trials", type=int, default=200,
                        help="n_trials for SMAC")
    parser.add_argument("--min_budget", type=int, default=5,
                        help="minimum budget for SMAC")
    parser.add_argument("--eta", type=int, default=3,
                        help="eta for Hyperband")
    args, extra_args = parser.parse_known_args()
    set_loglevel(args.log_level)
    experiment_file = args.experiment_file
    engine = Engine()
    engine.parse_experiment_from_file(experiment_file, extra_args=extra_args)
    engine.prepare_data()
    run = next(engine.runs())
    max_epochs = run.task.max_epochs
    optimizer_name = run.optimizer.name
    cores = run.engine.workers * run.engine.devices
    max_time_per_job = sbatch_time(run.engine.sbatch_args["time"], run.engine.sbatch_time_factor)
    devices = run.engine.devices
    outdir = run.engine.output_dir
    partition = run.engine.sbatch_args["partition"]
    del engine
    incumbent = run_smac(get_target_fn(extra_args, experiment_file), args, optimizer_name, max_epochs, outdir,
                         cores, max_time_per_job, devices, partition)
    print(incumbent)
