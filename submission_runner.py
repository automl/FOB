import argparse
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
import torch

import workloads
import submissions


def main(args: argparse.Namespace):
    torch.set_float32_matmul_precision('high') # TODO: check if gpu has tensor cores
    datasets_dir: Path = args.datasets
    if args.download:
        raise NotImplementedError("download on demand not implemented yet")
    workload = workloads.import_workload(args.workload)
    submission = submissions.import_submission(args.submission)

    data_module = workload.get_datamodule(datasets_dir)
    model = workload.get_model(submission.get_submission(args.hyperparameters))
    specs = workload.get_specs(model, data_module)
    trainer = L.Trainer(
        max_epochs=specs["max_epochs"],  # TODO: use max_steps instead?
        callbacks=[
            LearningRateMonitor()
        ],
        devices=1  # TODO: adjust according to workload
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="runs a single submission (optimizer and scheduler) on a single workload")
    parser.add_argument("--datasets", "-d", required=True, type=Path, help="path to all datasets (should be workload independent)")
    parser.add_argument("--download", default=False, action="store_true", help="download dataset if it does not exist")
    parser.add_argument("--checkpoints", "-c", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--workload", "-w", required=True, type=str, choices=workloads.workload_names())
    parser.add_argument("--submission", "-s", required=True, type=str, choices=submissions.submission_names())
    parser.add_argument("--hyperparameters", type=Path, help="Path to hyperparameters file")
    # TODO: hyperparameter, trial number, experiment name
    args = parser.parse_args()
    main(args)
