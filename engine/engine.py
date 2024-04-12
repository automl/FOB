from typing import Any, Callable, Iterable, Iterator, Optional
from pathlib import Path
import hashlib
import time
import torch
import yaml
from pandas import DataFrame
from lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.loggers import Logger, TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_warn
from evaluation import evaluation_path
from evaluation.plot import create_figure, dataframe_from_trials, get_output_file_path, save_files, set_plotstyle
from optimizers import Optimizer, optimizer_path, optimizer_names
from tasks import TaskModel, TaskDataModule, import_task, task_path, task_names
from .configs import EngineConfig, EvalConfig, OptimizerConfig, TaskConfig
from .callbacks import LogParamsAndGrads, PrintEpoch
from .grid_search import gridsearch
from .parser import YAMLParser
from .utils import AttributeDict, calculate_steps, findfirst, path_to_str_inside_dict, dict_differences, concatenate_dict_keys, precision_with_fallback, seconds_to_str, trainer_strategy, write_results


def engine_path() -> Path:
    return Path(__file__).resolve().parent


class Run():
    def __init__(
            self,
            config: dict[str, Any],
            default_config: dict[str, Any],
            task_key: str,
            optimizer_key: str,
            engine_key: str,
            eval_key: str,
            identifier_key: str
            ) -> None:
        self._config = config
        self._default_config = default_config
        self.task_key = task_key
        self.optimizer_key = optimizer_key
        self.engine_key = engine_key
        self.eval_key = eval_key
        self.identifier_key = identifier_key
        self._generate_configs()
        self._set_outpath()
        self._callbacks = AttributeDict({})

    def start(self):
        self._ensure_max_steps()
        self._ensure_resume_path()
        torch.set_float32_matmul_precision('high')
        seed_everything(self.engine.seed, workers=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.export_config()
        model, data_module = self.get_task()
        # TODO: test only correctness for last checkpoint
        if self.engine.train:
            trainer = self.get_trainer()
            self.train(trainer, model, data_module)
        if self.engine.test:
            tester = self.get_tester()
            self.test(tester, model, data_module)
            best_path = self.get_best_checkpoint()
            if best_path is not None:
                self.test(tester, model, data_module, Path(best_path))

    def train(self, trainer: Trainer, model: LightningModule, data_module: LightningDataModule):
        start_time = time.time()
        if self.engine.accelerator == "gpu" and torch.cuda.is_available():
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=True,
                enable_mem_efficient=(self.engine.optimize_memory or not self.engine.deterministic)
            ):
                trainer.fit(model, datamodule=data_module, ckpt_path=self.engine.resume)  # type: ignore
        else:
            trainer.fit(model, datamodule=data_module, ckpt_path=self.engine.resume)  # type: ignore
        end_time = time.time()
        train_time = int(end_time - start_time)
        rank_zero_info(f"Finished training in {seconds_to_str(train_time)}.")

    def test(self, tester: Trainer, model: LightningModule, data_module: LightningDataModule, ckpt: Optional[Path] = None):
        ckpt_path = self.engine.resume if ckpt is None else ckpt
        mode = "final" if ckpt_path is None or ckpt_path.stem.startswith("last") else "best"
        score = tester.test(model, datamodule=data_module, ckpt_path=ckpt_path)  # type: ignore
        write_results(score, self.run_dir / f"results_{mode}_model.json")

    def export_config(self):
        with open(self.run_dir / "config.yaml", "w", encoding="utf8") as f:
            yaml.safe_dump(path_to_str_inside_dict(self._config), f)

    def get_optimizer(self) -> Optimizer:
        return Optimizer(self.optimizer)

    def get_task(self) -> tuple[TaskModel, TaskDataModule]:
        task_module = import_task(self.task.name)
        return task_module.get_task(self.get_optimizer(), self.task)

    def get_datamodule(self) -> TaskDataModule:
        task_module = import_task(self.task.name)
        return task_module.get_datamodule(self.task)

    def get_callbacks(self) -> list[Callback]:
        if len(self._callbacks) < 1:
            self._init_callbacks()
        return list(self._callbacks.values())

    def get_loggers(self) -> list[Logger]:
        return [
            TensorBoardLogger(
                save_dir=self.run_dir,
                name="tb_logs"
            ),
            CSVLogger(
                save_dir=self.run_dir,
                name="csv_logs"
            )
        ]

    def get_trainer(self) -> Trainer:
        return Trainer(
            max_steps=self.engine.max_steps,
            logger=self.get_loggers(),
            callbacks=self.get_callbacks(),
            devices=self.engine.devices,
            strategy=trainer_strategy(self.engine.devices),
            enable_progress_bar=(not self.engine.silent),
            deterministic=self.engine.deterministic,
            detect_anomaly=self.engine.detect_anomaly,
            gradient_clip_val=self.engine.gradient_clip_val,
            gradient_clip_algorithm=self.engine.gradient_clip_alg,
            precision=precision_with_fallback(self.engine.precision),  # type: ignore
            accelerator=self.engine.accelerator
        )

    def get_tester(self) -> Trainer:
        return Trainer(
            devices=1,
            enable_progress_bar=(not self.engine.silent),
            deterministic=self.engine.deterministic,
            precision=precision_with_fallback(self.engine.precision),  # type: ignore
            accelerator=self.engine.accelerator
        )

    def get_best_checkpoint(self) -> Optional[Path]:
        model_checkpoint = self._callbacks.get("model_checkpoint", None)
        if model_checkpoint is not None:
            model_checkpoint = model_checkpoint.best_model_path
        if model_checkpoint is None:
            available_checkpoints = self.get_available_checkpoints()
            model_checkpoint = findfirst(lambda x: x.stem.startswith("best"), available_checkpoints)
        return model_checkpoint

    def get_available_checkpoints(self) -> list[Path]:
        if self.checkpoint_dir.exists():
            return list(filter(lambda x: x.suffix == ".ckpt", self.checkpoint_dir.iterdir()))
        return []

    def _ensure_resume_path(self):
        if isinstance(self.engine.resume, Path):
            pass
        elif isinstance(self.engine.resume, bool):
            resume_path = None
            if self.engine.resume:
                available_checkpoints = self.get_available_checkpoints()
                if len(available_checkpoints) < 1:
                    rank_zero_warn("engine.resume=True but no checkpoint was found. Starting run from scratch.")
                else:
                    resume_path = findfirst(lambda x: x.stem == "last", available_checkpoints)
            self._config[self.engine_key]["resume"] = resume_path
            self._generate_configs()
        else:
            raise TypeError(f"Unsupportet type for 'resume', got {type(self.engine.resume)=}.")

    def _ensure_max_steps(self):
        if self.task.max_steps is None:
            max_steps = self._calc_max_steps()
            self._config[self.task_key]["max_steps"] = max_steps
            if self._default_config[self.task_key]["max_steps"] is None:
                self._default_config[self.task_key]["max_steps"] = max_steps
            self._generate_configs()
            rank_zero_info(f"'max_steps' not set explicitly, using {max_steps=} (calculated from " +
            f"max_epochs={self.task.max_epochs}, batch_size={self.task.batch_size}, devices={self.engine.devices})")

    def _calc_max_steps(self) -> int:
        dm = self.get_datamodule()
        dm.setup("fit")
        train_samples = len(dm.data_train)
        return calculate_steps(self.task.max_epochs, train_samples, self.engine.devices, self.task.batch_size)

    def _init_callbacks(self):
        self._callbacks["model_checkpoint"] = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename="best-{epoch}-{step}",
            monitor=self.task.target_metric,
            mode=self.task.target_metric_mode,
            save_last=True
        )
        self._callbacks["lr_monitor"] = LearningRateMonitor(logging_interval=self.optimizer.lr_interval)
        self._callbacks["extra"] = LogParamsAndGrads(
            log_gradient=self.engine.log_extra,
            log_params=self.engine.log_extra,
            log_quantiles=self.engine.log_extra,
            log_every_n_steps=100  # maybe add arg for this?
        )
        self._callbacks["print_epoch"] = PrintEpoch(self.engine.silent)

    def outpath_exclude_keys(self) -> list[str]:
        return [
            self.eval_key,
            "output_dir_name"
        ]

    def _set_outpath(self):
        self._ensure_max_steps()
        base: Path = self.engine.output_dir / self.task.output_dir_name / self.optimizer.output_dir_name
        exclude_keys = self.outpath_exclude_keys()
        exclude_keys += self.engine.outpath_irrelevant_engine_keys()
        diffs = concatenate_dict_keys(dict_differences(self._config, self._default_config), exclude_keys=exclude_keys)
        run_dir = ",".join(f"{k}={str(v)}" for k, v in sorted(diffs.items())) if diffs else "default"
        if len(run_dir) > 254:  # max file name length
            hashdir = hashlib.md5(run_dir.encode()).hexdigest()
            rank_zero_warn(f"folder name {run_dir} is too long, using {hashdir} instead.")
            run_dir = hashdir
        self.run_dir = base / run_dir
        self.checkpoint_dir = self.run_dir / "checkpoints"

    def _generate_configs(self):
        self.engine = EngineConfig(self._config, self.task_key, self.engine_key)
        self.optimizer = OptimizerConfig(self._config, self.optimizer_key, self.task_key, self.identifier_key)
        self.task = TaskConfig(self._config, self.task_key, self.engine_key, self.identifier_key)
        self.evaluation = EvalConfig(
            self._config,
            eval_key=self.eval_key,
            optimizer_key=self.optimizer_key,
            identifier_key=self.identifier_key,
            ignore_keys=self.engine.outpath_irrelevant_engine_keys(prefix=f"{self.engine_key}.")
        )


class Engine():
    def __init__(self) -> None:
        self._runs = []
        self._defaults = []
        self.task_key = "task"
        self.optimizer_key = "optimizer"
        self.engine_key = "engine"
        self.eval_key = "evaluation"
        self.identifier_key = "name"
        self.default_file_name = "default.yaml"
        self.parser = YAMLParser()

    def run_experiment(self):
        # TODO: early stopping and detect_anomaly
        assert len(self._runs) > 0, "No runs in experiment, make sure to call 'parse_experiment' first."
        scheduler = self._runs[0][self.engine_key]["run_scheduler"]
        assert all(map(lambda x: x[self.engine_key]["run_scheduler"] == scheduler, self._runs)), \
            "You cannot perform gridsearch on 'run_scheduler'."
        if scheduler == "sequential":
            for i, run in enumerate(self.runs(), start=1):
                rank_zero_info(f"Starting run {i}/{len(self._runs)}.")
                run.start()
        elif scheduler.startswith("single"):
            n = int(scheduler.rsplit(":", 1)[-1])
            for i, run in enumerate(self.runs(), start=1):
                if i == n:
                    rank_zero_info(f"Starting run {i}/{len(self._runs)}.")
                    run.start()
        # TODO: support slurm
        elif scheduler == "slurm_array":
            raise NotImplementedError("Slurm scheduler not implemented yet.")
        else:
            raise ValueError(f"Unsupported run_scheduler: {scheduler=}.")

    def parse_experiment_from_file(self, file: Path, extra_args: Iterable[str] = tuple()):
        searchspace: dict[str, Any] = self.parser.parse_yaml(file)
        self.parse_experiment(searchspace, extra_args)

    def parse_experiment(self, searchspace: dict[str, Any], extra_args: Iterable[str] = tuple()):
        self.parser.parse_args_into_searchspace(searchspace, extra_args)
        self._named_dicts_to_list(
            searchspace,
            [self.optimizer_key, self.task_key],
            [optimizer_names(), task_names()]
        )
        # exclude plotting from gridsearch
        if self.eval_key in searchspace:
            eval_config = searchspace.pop(self.eval_key)
        else:
            eval_config = {}
        self._runs = gridsearch(searchspace)
        for run in self._runs:
            run[self.eval_key] = eval_config
        self._fill_runs_from_default(self._runs)
        self._fill_defaults()

    def runs(self) -> Iterator[Run]:
        for config, default_config in zip(self._runs, self._defaults):
            run = Run(
                config,
                default_config,
                self.task_key,
                self.optimizer_key,
                self.engine_key,
                self.eval_key,
                self.identifier_key
            )
            yield run

    def plot(self):
        return self.plot_lazy()

    def plot_lazy(self):
        config = next(self.runs()).evaluation
        set_plotstyle(config)
        trials = list(map(lambda x: Path(x.run_dir), self.runs()))
        df = dataframe_from_trials(trials, config)

        dfs: list[DataFrame] = [group for _, group in df.groupby(config.column_split_key)]
        fig, axs = create_figure(dfs, config)

        output_file_path = get_output_file_path(dfs, config)
        save_files(dfs, output_file_path, config)

    def plot_clean(self):
        # TODO: create dataframes in engine and plot them
        raise NotImplementedError("The implementation of this is trivial and left as an exercise for the reader.")

    def _named_dicts_to_list(self, searchspace: dict[str, Any], keys: list[str], valid_options: list[list[str]]):
        assert len(keys) == len(valid_options)
        for key, opts in zip(keys, valid_options):
            if key not in searchspace:
                continue
            if isinstance(searchspace[key], dict) and all(name in opts for name in searchspace[key]):
                searchspace[key] = [cfg | {self.identifier_key: name} for name, cfg in searchspace[key].items()]

    def _fill_defaults(self):
        self._defaults = []
        for run in self._runs:
            default_cfg = {
                k: {self.identifier_key: run[k][self.identifier_key]}
                for k in [self.task_key, self.optimizer_key]
            }
            self._defaults.append(default_cfg)
        self._fill_runs_from_default(self._defaults)

    def _fill_runs_from_default(self, runs: list[dict[str, Any]]):
        for i, _ in enumerate(runs):
            # order from higher to lower in hierarchy
            runs[i] = self._fill_named_from_default(runs[i], self.task_key, task_path)
            runs[i] = self._fill_named_from_default(runs[i], self.optimizer_key, optimizer_path)
            runs[i] = self._fill_unnamed_from_default(runs[i], engine_path)
            runs[i] = self._fill_unnamed_from_default(runs[i], evaluation_path)

    def _fill_unnamed_from_default(self, experiment: dict[str, Any], unnamed_root: Callable) -> dict[str, Any]:
        default_path: Path = unnamed_root() / self.default_file_name
        default_config = self.parser.parse_yaml(default_path)
        self.parser.merge_dicts_hierarchical(default_config, experiment)
        return default_config

    def _fill_named_from_default(self, experiment: dict[str, Any], key: str, named_root: Callable) -> dict[str, Any]:
        self._argcheck_named(experiment, key, self.identifier_key)
        named = experiment[key]
        if isinstance(named, dict):
            named = named[self.identifier_key]
        else:
            experiment[key] = {self.identifier_key: named}
        default_path: Path = named_root(named) / self.default_file_name
        default_config = self.parser.parse_yaml(default_path)
        self.parser.merge_dicts_hierarchical(default_config, experiment)
        return default_config

    def _argcheck_named(self, experiment: dict[str, Any], key: str, identifier: str):
        assert key in experiment, f"You did not provide any {key}."
        assert isinstance(experiment[key], str) or identifier in experiment[key], \
            f"Unknown {key}, either specify only a string or provide a key '{identifier}'"
