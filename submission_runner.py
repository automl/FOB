import argparse
from pathlib import Path
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import torch

from runtime import RuntimeArgs
from runtime.callbacks import LogParamsAndGrads, PrintEpoch
from runtime.utils import some, trainer_strategy, begin_timeout, write_results

import workloads
from workloads import WorkloadModel, WorkloadDataModule
import submissions


def run_trial(runtime_args: RuntimeArgs):
    torch.set_float32_matmul_precision('high')  # TODO: check if gpu has tensor cores
    seed_everything(runtime_args.seed, workers=True)
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
        filename="best-{epoch}-{step}",
        monitor=specs.target_metric,
        mode=specs.target_metric_mode,
        save_last=True
    )
    max_epochs = specs.max_epochs if specs.max_steps is None else None
    max_steps = some(specs.max_steps, default=-1)
    devices = some(runtime_args.devices, default=specs.devices)
    trainer = Trainer(
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
            ),
            PrintEpoch(runtime_args.silent)  # TODO: verbosity level
        ],
        devices=devices,
        strategy=trainer_strategy(devices),
        enable_progress_bar=(not runtime_args.silent),
        deterministic="warn" if runtime_args.deterministic else False,
        precision="bf16-mixed"
    )
    tester = Trainer(
        callbacks=[*(workload.get_callbacks())],
        devices=1,
        enable_progress_bar=(not runtime_args.silent),
        deterministic="warn" if runtime_args.deterministic else False,
        precision="bf16-mixed"
    )
    if runtime_args.test_only:
        ckpt_path = runtime_args.resume
        mode = "final" if ckpt_path is None or ckpt_path.stem.startswith("last") else "best"
        score = tester.test(model, datamodule=data_module, ckpt_path=ckpt_path)
        write_results(score, runtime_args.output_dir / f"results_{mode}_model.json")
    else:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=(runtime_args.optimize_memory or not runtime_args.deterministic)
        ):
            trainer.fit(model, datamodule=data_module, ckpt_path=runtime_args.resume)
        final_score = tester.test(model, datamodule=data_module)
        best_score = tester.test(model, datamodule=data_module, ckpt_path=model_checkpoint.best_model_path)
        write_results(final_score, runtime_args.output_dir / "results_final_model.json")
        write_results(best_score, runtime_args.output_dir / "results_best_model.json")


def main(args: argparse.Namespace):
    for trial in range(args.start_trial, args.start_trial + args.trials):
        print(f"Running trial {trial}.")
        runtime_args = RuntimeArgs(args)
        run_trial(runtime_args)

    if args.send_timeout:
        print("submission_runner.py finished! Setting timeout of 10 seconds, as tqdm sometimes is stuck\n")
        begin_timeout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="runs a single submission (optimizer and scheduler) on a single workload"
    )
    parser.add_argument("--data_dir", "-d", required=True, type=Path,
                        help="path to all datasets (should be workload independent)")
    parser.add_argument("--download", default=False, action="store_true",
                        help="download dataset if it does not exist")
    parser.add_argument("--output", "-o", type=Path,
                        help="where to store benchmark results, default: ./experiments")
    parser.add_argument("--workload", "-w", required=True, type=str, choices=workloads.workload_names())
    parser.add_argument("--submission", "-s", required=True, type=str, choices=submissions.submission_names())
    parser.add_argument("--hyperparameters", type=Path,
                        help="path to hyperparameters file or a directory of files")
    parser.add_argument("--resume", "-r", type=Path,
                        help="path to checkpoint file from which to resume")
    parser.add_argument("--workers", type=int,
                        help="number of parallelism used for loading data, default: all available")
    parser.add_argument("--trials", type=int, default=1,
                        help="number of trials, default: 1")
    parser.add_argument("--start_trial", type=int, default=0,
                        help="the number which is used for the first trial, default: 0")
    parser.add_argument("--start_hyperparameter", type=int, default=0,
                        help="the index of the first hyperparameter to run, default: 0")
    parser.add_argument("--seed", type=int, default=42,
                        help="the seed to use for the experiment if strategy is not 'random', default: 42")
    parser.add_argument("--seed_mode", type=str, default="none", choices=["fixed", "increment", "random"],
                        help="the strategy for choosing seeds, default: 'increment' on single hyperparameter, 'fixed' when using a search space")
    parser.add_argument("--devices", type=int,
                        help="overrides the predefined number of devices of the workload")
    parser.add_argument("--silent", action="store_true",
                        help="disable progress bars")
    parser.add_argument("--log_extra", action="store_true",
                        help="log training behavior like gradients etc")
    parser.add_argument("--send_timeout", action="store_true",
                        help="send a timeout after finishing this script (if you have problems with tqdm being stuck)")
    parser.add_argument("--test_only", action="store_true",
                        help="Skip training and only evaluate the model (provide checkpoint with the '--resume' arg).")
    parser.add_argument("--deterministic", type=bool, default=True,
                        help="Whether to use deterministic algorithms if possible.")
    parser.add_argument("--optimize_memory", action="store_true",
                        help="Use memory efficient attention, which is non-deterministic.")
    args = parser.parse_args()
    main(args)
