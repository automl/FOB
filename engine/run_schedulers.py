from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Optional
from slurmpy import Slurm
import yaml

from engine.run import Run
from engine.utils import log_info, seconds_to_str, some, str_to_seconds


# TODO: SLURM: plot only after all runs are finished


def argcheck_allequal_engine(runs: list[Run], keys: list[str]) -> bool:
    first = runs[0]
    for key in keys:
        if not all(run.engine[key] == first.engine[key] for run in runs[1:]):
            return False
    return True


def export_experiment(run: Run, experiment: dict[str, Any]) -> Path:
    run.run_dir.mkdir(parents=True, exist_ok=True)
    outfile = run.run_dir / "experiment.yaml"
    with open(outfile, "w", encoding="utf8") as f:
        yaml.safe_dump(experiment, f)
    return outfile


def process_args(args: dict[str, str], run: Run) -> None:
    if "time" in args:
        time = args["time"]
        seconds = str_to_seconds(time) if isinstance(time, str) else time
        args["time"] = seconds_to_str(int(run.engine.sbatch_time_factor * seconds))
    if "gres" not in args and "gpus" not in args:
        args["gres"] = f"gpu:{run.engine.devices}"
    if not any(k.startswith("ntasks") for k in args):
        args["ntasks-per-node"] = str(run.engine.devices)
    if not any(k.startswith("cpus") for k in args):
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
        scripts_dir=str(some(scripts_dir, run.engine.save_sbatch_scripts, default="fob-slurm-scripts")),
        bash_strict=False  # TODO: maybe add arg or just remove 'nounset'
    )


def run_slurm(command: str, run: Run, args: dict[str, str], log_dir: Path):
    if run.engine.save_sbatch_scripts is None:
        with TemporaryDirectory() as tmpdir:
            s = get_slurm(run, args, log_dir, Path(tmpdir).resolve())
            s.run(command)
    else:
        s = get_slurm(run, args, log_dir)
        s.run(command)


def slurm_array(runs: list[Run], run_script: Path, experiment: dict[str, Any]) -> None:
    equal_req = ["devices", "workers", "sbatch_args", "slurm_log_dir", "sbatch_script_template", "run_scheduler"]
    ok = argcheck_allequal_engine(runs, equal_req)
    if not ok:
        raise ValueError(f"All runs must have the same values for {', '.join(map(lambda s: 'engine.' + s, equal_req))} when using 'engine.run_scheduler=slurm_array'")
    run = runs[0]  # all runs have the same args
    args = run.engine.sbatch_args
    log_dir = some(run.engine.slurm_log_dir, default=run.engine.output_dir / "slurm_logs")
    if "array" not in args:
        args["array"] = f"1-{len(runs)}"
    process_args(args, run)
    experiment_file = [export_experiment(run, experiment).resolve() for run in runs][0]
    command = get_command(run_script, experiment_file, "$SLURM_ARRAY_TASK_ID")
    command = wrap_template(run.engine.sbatch_script_template, command)
    run_slurm(command, run, args, log_dir)


def slurm_jobs(runs: Iterable[Run], run_script: Path, experiment: dict[str, Any]) -> None:
    for i, run in enumerate(runs, start=1):
        args = run.engine.sbatch_args
        process_args(args, run)
        log_dir = some(run.engine.slurm_log_dir, default=run.run_dir / "slurm_logs")
        experiment_file = export_experiment(run, experiment).resolve()
        command = get_command(run_script, experiment_file, str(i))
        command = wrap_template(run.engine.sbatch_script_template, command)
        run_slurm(command, run, args, log_dir)


def sequential(runs: Iterable[Run], n_runs: int, experiment: dict[str, Any]):
    for i, run in enumerate(runs, start=1):
        log_info(f"Starting run {i}/{n_runs}.")
        export_experiment(run, experiment)
        try:
            run.start()
        except RuntimeError as e:  # detect_anomaly raises RuntimeError
            log_info(f"Run {i}/{n_runs} failed with {e}.")
