from pathlib import Path
from typing import Iterable
from slurmpy import Slurm

from engine.run import Run
from engine.utils import some


# TODO: option to save or discard sbatch scripts
# TODO: bash script template for setup etc.

def argcheck_allequal_engine(runs: list[Run], keys: list[str]) -> bool:
    first = runs[0]
    for key in keys:
        if not all(run.engine[key] == first.engine[key] for run in runs[1:]):
            return False
    return True


def ensure_args(args: dict[str, str], run: Run) -> None:
    if not "gres" in args:
        args["gres"] = f"gpu:{run.engine.devices}"
    if not any(k in args for k in ["ntasks", "ntasks-per-node"]):
        args["ntasks"] = str(run.engine.devices)
    if not any(k.startswith("cpu") for k in args):
        args["cpus-per-task"] = str(run.engine.workers)


def slurm_array(runs: list[Run], run_script: Path, experiment_file: Path) -> None:
    ok = argcheck_allequal_engine(runs, ["devices", "workers", "sbatch_args", "run_scheduler"])
    if not ok:
        raise ValueError("All runs must have the same values for 'engine.devices', 'engine.workers', 'engine.sbatch_args', and 'engine.run_scheduler' when using 'engine.run_scheduler=slurm_array'")
    n_runs = len(runs)
    run = runs[0]
    args = run.engine.sbatch_args
    log_dir = some(run.engine.slurm_log_dir, default=run.engine.output_dir / "slurm_logs")
    if not "array" in args:
        args["array"] = f"1-{n_runs}"
    ensure_args(args, run)
    s = Slurm(f"FOB-{run.task.name}-{run.optimizer.name}", args, log_dir=str(log_dir.resolve()))
    command = f"""srun python {run_script} {experiment_file} "engine.run_scheduler=single:$SLURM_ARRAY_TASK_ID"
    """
    s.run(command)

def slurm_jobs(runs: Iterable[Run], run_script: Path, experiment_file: Path) -> None:
    # TODO: do not pass experiment file to sbatch calls, instead pass command line args
    for i, run in enumerate(runs):
        args = run.engine.sbatch_args
        ensure_args(args, run)
        log_dir = some(run.engine.slurm_log_dir, default=run.run_dir / "slurm_logs")
        s = Slurm(f"FOB-{run.task.name}-{run.optimizer.name}", args, log_dir=str(log_dir.resolve()))
        command = f"""srun python {run_script} {experiment_file} "engine.run_scheduler=single:{i}"
        """
        s.run(command)
