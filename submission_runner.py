import argparse
from pathlib import Path
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import torch
import sys

from runtime.runtime import Runtime, Run
from runtime.callbacks import LogParamsAndGrads, PrintEpoch
from runtime.utils import some, trainer_strategy, begin_timeout, write_results


def run_trial(run: Run):
    torch.set_float32_matmul_precision('high')  # TODO: check if gpu has tensor cores
    if not torch.cuda.is_bf16_supported():
        print("Warning: GPU does not support bfloat16, using float16. Results can be different!", file=sys.stderr)
    seed_everything(run.runtime.seed, workers=True)
    run.export_config()
    model, data_module = run.get_workload()
    model_checkpoint = ModelCheckpoint(
        dirpath=run.output_dir / "checkpoints",
        filename="best-{epoch}-{step}",
        monitor=run.workload.target_metric,
        mode=run.workload.target_metric_mode,
        save_last=True
    )
    max_epochs = run.workload.max_epochs if run.workload.max_steps is None else None
    max_steps = some(run.workload.max_steps, run.workload.max_steps, default=-1)
    devices = some(run.runtime.devices, default=run.runtime.devices)
    trainer = Trainer(
        max_epochs=max_epochs,
        max_steps=max_steps,
        logger=[
            TensorBoardLogger(
                save_dir=run.output_dir,
                name="tb_logs"
            ),
            CSVLogger(
                save_dir=run.output_dir,
                name="csv_logs"
            )
        ],
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            model_checkpoint,
            LogParamsAndGrads(
                log_gradient=run.runtime.log_extra,
                log_params=run.runtime.log_extra,
                log_quantiles=run.runtime.log_extra,
                log_every_n_steps=100  # maybe add arg for this?
            ),
            PrintEpoch(run.runtime.silent)  # TODO: verbosity level
        ],
        devices=devices,
        strategy=trainer_strategy(devices),
        enable_progress_bar=(not run.runtime.silent),
        deterministic="warn" if run.runtime.deterministic else False,
        precision="bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
    )
    tester = Trainer(
        devices=1,
        enable_progress_bar=(not run.runtime.silent),
        deterministic="warn" if run.runtime.deterministic else False,
        precision="bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
    )
    if run.runtime.test_only:
        ckpt_path = run.runtime.resume
        mode = "final" if ckpt_path is None or ckpt_path.stem.startswith("last") else "best"
        score = tester.test(model, datamodule=data_module, ckpt_path=ckpt_path)
        write_results(score, run.runtime.output_dir / f"results_{mode}_model.json")
    else:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=(run.runtime.optimize_memory or not run.runtime.deterministic)
        ):
            trainer.fit(model, datamodule=data_module, ckpt_path=run.runtime.resume)
        final_score = tester.test(model, datamodule=data_module)
        best_score = tester.test(model, datamodule=data_module, ckpt_path=model_checkpoint.best_model_path)
        write_results(final_score, run.output_dir / "results_final_model.json")
        write_results(best_score, run.output_dir / "results_best_model.json")


def main(args: argparse.Namespace, extra_args: list[str]):
    runtime = Runtime()
    runtime.parse_experiment(args.experiment_file, extra_args=extra_args)
    runs = runtime.runs()
    for i, run in enumerate(runs):
        print(f"Starting run {i + 1}/{len(runs)}.")
        run_trial(run)

    if args.send_timeout:
        print("submission_runner.py finished! Setting timeout of 10 seconds, as tqdm sometimes is stuck\n")
        begin_timeout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="runs an experiment specified by a file"
    )
    parser.add_argument("experiment_file", type=Path,
                        help="The yaml file specifying the experiment.")
    parser.add_argument("--send_timeout", action="store_true",
                        help="send a timeout after finishing this script (if you have problems with tqdm being stuck)")
    args, extra_args = parser.parse_known_args()
    main(args, extra_args)
