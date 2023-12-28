import argparse
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import json

from bob.runtime import RuntimeArgs

import workloads
from workloads import WorkloadModel, WorkloadDataModule
import submissions


def main(runtime_args: RuntimeArgs):
    torch.set_float32_matmul_precision('high') # TODO: check if gpu has tensor cores
    workload = workloads.import_workload(runtime_args.workload_name)
    submission = submissions.import_submission(runtime_args.submission_name)

    wl: tuple[WorkloadModel, WorkloadDataModule] = workload.get_workload(
        submission.get_submission(runtime_args),
        runtime_args
    )
    model, data_module = wl
    specs = model.get_specs()
    model_checkpoint = ModelCheckpoint(
        dirpath=runtime_args.checkpoint_dir,
        monitor=specs.target_metric,
        mode=specs.target_metric_mode
    )
    trainer = L.Trainer(
        max_epochs=specs.max_epochs,
        callbacks=[
            *(workload.get_callbacks()),
            LearningRateMonitor(),
            model_checkpoint
        ],
        devices=specs.devices
    )
    trainer.fit(model, datamodule=data_module)
    final_score = trainer.test(model, datamodule=data_module)
    best_score = trainer.test(model, datamodule=data_module, ckpt_path=model_checkpoint.best_model_path)
    with open(runtime_args.output_dir / "results_final_model.json", "w", encoding="utf8") as f:
        json.dump(final_score, f)
    with open(runtime_args.output_dir / "results_best_model.json", "w", encoding="utf8") as f:
        json.dump(best_score, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="runs a single submission (optimizer and scheduler) on a single workload"
    )
    parser.add_argument("--data_dir", "-d", required=True, type=Path, \
                        help="path to all datasets (should be workload independent)")
    parser.add_argument("--download", default=False, action="store_true", \
                        help="download dataset if it does not exist")
    parser.add_argument("--checkpoints", "-c", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--workload", "-w", required=True, type=str, choices=workloads.workload_names())
    parser.add_argument("--submission", "-s", required=True, type=str, choices=submissions.submission_names())
    parser.add_argument("--hyperparameters", type=Path, \
                        help="path to hyperparameters file")
    parser.add_argument("--workers", type=int, \
                        help="number of parallelism used for loading data, default: all available")
    # TODO: hyperparameter, trial number, experiment name
    args = parser.parse_args()
    runtime_args = RuntimeArgs(args)
    main(runtime_args)
