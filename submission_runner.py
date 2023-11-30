import argparse
from pathlib import Path
import lightning as L

import workloads
import submissions


def main(args: argparse.Namespace):
    datasets_dir: Path = args.datasets
    if args.download:
        raise NotImplementedError("download on demand not implemented yet")
    workload = workloads.import_workload(args.workload)
    submission = submissions.import_submission(args.submission)

    data_module = workload.get_datamodule(datasets_dir)
    model = workload.get_model(submission.configure_optimizers)
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="runs a single submission (optimizer and scheduler) on a single workload")
    parser.add_argument("--datasets", "-d", required=True, type=Path, help="path to all datasets (should be workload independent)")
    parser.add_argument("--download", default=False, action="store_true", help="download dataset if it does not exist")
    parser.add_argument("--checkpoints", "-c", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--workload", "-w", required=True, type=str, choices=workloads.workload_names())
    parser.add_argument("--submission", "-s", required=True, type=str, choices=submissions.submission_names())
    # TODO: hyperparameter, trial number, experiment name
    args = parser.parse_args()
    main(args)
