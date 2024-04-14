from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Optional
from slurmpy import Slurm

from engine.run import Run
from engine.utils import some


# TODO: time multiplier for slurm jobs
# TODO: make this work for extra args


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


def wrap_template(template_path: Optional[Path], command: str, placeholder: str = "__FOB_COMMAND__") -> str:
    if template_path is not None:
        with open(template_path, "r", encoding="utf8") as f:
            template = f.read()
            if placeholder in template:
                command = template.replace(placeholder, command)
            else:
                command = f"{template}\n{command}\n"
    return command


def get_command(run_script: Path, experiment_file: Path, index: str) -> str:
    return f"""srun python {run_script} {experiment_file} "engine.run_scheduler=single:{index}" """


def get_slurm(run: Run, args: dict[str, str], log_dir: Path, scripts_dir: Optional[Path] = None) -> Slurm:
    return Slurm(
        f"FOB-{run.task.name}-{run.optimizer.name}",
        args,
        log_dir=str(log_dir.resolve()),
        scripts_dir=str(some(scripts_dir, run.engine.save_sbatch_scripts, default="fob-slurm-scripts"))
    )


def run_slurm(command: str, run: Run, args: dict[str, str], log_dir: Path):
    if run.engine.save_sbatch_scripts is None:
        with TemporaryDirectory() as tmpdir:
            s = get_slurm(run, args, log_dir, Path(tmpdir).resolve())
            s.run(command)
    else:
        s = get_slurm(run, args, log_dir)
        s.run(command)


def slurm_array(runs: list[Run], run_script: Path, experiment_file: Path) -> None:
    equal_req = ["devices", "workers", "sbatch_args", "slurm_log_dir", "sbatch_script_template", "run_scheduler"]
    ok = argcheck_allequal_engine(runs, equal_req)
    if not ok:
        raise ValueError(f"All runs must have the same values for {', '.join(map(lambda s: 'engine.' + s, equal_req))} when using 'engine.run_scheduler=slurm_array'")
    run = runs[0]  # all runs have the same args
    args = run.engine.sbatch_args
    log_dir = some(run.engine.slurm_log_dir, default=run.engine.output_dir / "slurm_logs")
    if not "array" in args:
        args["array"] = f"1-{len(runs)}"
    ensure_args(args, run)
    command = get_command(run_script, experiment_file, "$SLURM_ARRAY_TASK_ID")
    command = wrap_template(run.engine.sbatch_script_template, command)
    run_slurm(command, run, args, log_dir)


def slurm_jobs(runs: Iterable[Run], run_script: Path, experiment_file: Path) -> None:
    # TODO: do not pass experiment file to sbatch calls, instead pass command line args
    for i, run in enumerate(runs, start=1):
        args = run.engine.sbatch_args
        ensure_args(args, run)
        log_dir = some(run.engine.slurm_log_dir, default=run.run_dir / "slurm_logs")
        command = get_command(run_script, experiment_file, str(i))
        command = wrap_template(run.engine.sbatch_script_template, command)
        run_slurm(command, run, args, log_dir)
