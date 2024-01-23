import argparse
import json
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import torch

from runtime import RuntimeArgs
from runtime.callbacks import LogParamsAndGrads
from runtime.utils import some, trainer_strategy

import workloads
from workloads import WorkloadModel, WorkloadDataModule
import submissions


def run_trial(runtime_args: RuntimeArgs):
    torch.set_float32_matmul_precision('high') # TODO: check if gpu has tensor cores
    L.seed_everything(runtime_args.seed)
    runtime_args.export_settings()
    workload = workloads.import_workload(runtime_args.workload_name)
    submission = submissions.import_submission(runtime_args.submission_name)

    wl: tuple[WorkloadModel, WorkloadDataModule] = workload.get_workload(
        submission.get_submission(runtime_args),
        runtime_args
    )
    model, data_module = wl
    specs = model.get_specs()
    specs.export_settings(runtime_args.output_dir)
    model_checkpoint = ModelCheckpoint(
        dirpath=runtime_args.checkpoint_dir,
        monitor=specs.target_metric,
        mode=specs.target_metric_mode
    )
    max_epochs = specs.max_epochs if specs.max_steps is None else None
    max_steps = some(specs.max_steps, default=-1)
    devices = some(runtime_args.devices, default=specs.devices)
    n_devices = devices if isinstance(devices, int) else len(devices)
    trainer = L.Trainer(
        max_epochs=max_epochs,
        max_steps=max_steps,
        logger=[
            TensorBoardLogger(
                save_dir=runtime_args.output_dir,
                name="tb_logs"
            ),
            CSVLogger(
                save_dir=runtime_args.output_dir,
                name="csv_logs"
            )
        ],
        callbacks=[
            *(workload.get_callbacks()),
            LearningRateMonitor(logging_interval="step"),
            model_checkpoint,
            LogParamsAndGrads(
                log_gradient=runtime_args.log_extra,
                log_params=runtime_args.log_extra,
                log_quantiles=runtime_args.log_extra,
                log_every_n_steps=100  # maybe add arg for this?
            )
        ],
        devices=devices,
        strategy=trainer_strategy(devices),
        enable_progress_bar=(not runtime_args.silent)
    )
    trainer.fit(model, datamodule=data_module, ckpt_path=runtime_args.resume)
    final_score = trainer.test(model, datamodule=data_module)
    best_score = trainer.test(model, datamodule=data_module, ckpt_path=model_checkpoint.best_model_path)
    with open(runtime_args.output_dir / "results_final_model.json", "w", encoding="utf8") as f:
        json.dump(final_score, f, indent=4)
    with open(runtime_args.output_dir / "results_best_model.json", "w", encoding="utf8") as f:
        json.dump(best_score, f, indent=4)


def main(args: argparse.Namespace):
    for trial in range(args.start_trial, args.start_trial + args.trials):
        print(f"Running trial {trial}.")
        runtime_args = RuntimeArgs(args)
        run_trial(runtime_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="runs a single submission (optimizer and scheduler) on a single workload"
    )
    parser.add_argument("--data_dir", "-d", required=True, type=Path, \
                        help="path to all datasets (should be workload independent)")
    parser.add_argument("--download", default=False, action="store_true", \
                        help="download dataset if it does not exist")
    parser.add_argument("--output", "-o", type=Path, \
                        help="where to store benchmark results, default: ./experiments")
    parser.add_argument("--workload", "-w", required=True, type=str, choices=workloads.workload_names())
    parser.add_argument("--submission", "-s", required=True, type=str, choices=submissions.submission_names())
    parser.add_argument("--hyperparameters", type=Path, \
                        help="path to hyperparameters file or a directory of files")
    parser.add_argument("--resume", "-r", type=Path, \
                        help="path to checkpoint file from which to resume")
    parser.add_argument("--workers", type=int, \
                        help="number of parallelism used for loading data, default: all available")
    parser.add_argument("--trials", type=int, default=1, \
                        help="number of trials, default: 1")
    parser.add_argument("--start_trial", type=int, default=0, \
                        help="the index of the first trial to run, default: 0")
    parser.add_argument("--seed", type=int, default=42, \
                        help="the seed to use for the experiment if strategy is not 'random', default: 42")
    parser.add_argument("--seed_mode", type=str, default="increment", \
                        choices=["fixed", "increment", "random"], \
                        help="the strategy for choosing seeds, default: 'increment'")
    parser.add_argument("--devices", type=int, \
                        help="overrides the predefined number of devices of the workload")
    parser.add_argument("--silent", action="store_true", \
                        help="disable progress bars")
    parser.add_argument("--log_extra", action="store_true", \
                        help="log training behavior like gradients etc")
    args = parser.parse_args()
    main(args)
