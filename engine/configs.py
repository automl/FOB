from pathlib import Path
from typing import Any, Literal, Optional
from .utils import AttributeDict, convert_type_inside_dict, some, wrap_list


class BaseConfig(AttributeDict):
    def __init__(self, config: dict):
        super().__init__(convert_type_inside_dict(config, dict, AttributeDict))


class NamedConfig(BaseConfig):
    def __init__(
            self,
            config: dict[str, Any],
            identifier_key: str = "name",
            outdir_key: str = "output_dir_name"
            ) -> None:
        super().__init__(config)
        self.name = config[identifier_key]
        self.output_dir_name = config.get(outdir_key, self.name)


class OptimizerConfig(NamedConfig):
    def __init__(
            self,
            config: dict[str, Any],
            optimizer_key: str,
            task_key: str,
            identifier_key: str = "name",
            outdir_key: str = "output_dir_name"
            ) -> None:
        cfg = dict(config[optimizer_key])
        self.lr_interval: Literal["step", "epoch"] = cfg.get("lr_interval", "step")
        self.max_steps: int = config[task_key].get("max_steps", None)
        self.max_epochs: int = config[task_key]["max_epochs"]
        cfg["max_steps"] = self.max_steps
        cfg["max_epochs"] = self.max_epochs
        super().__init__(cfg, identifier_key, outdir_key)


class TaskConfig(NamedConfig):
    def __init__(
            self,
            config: dict[str, Any],
            task_key: str,
            engine_key: str,
            identifier_key: str = "name",
            outdir_key: str = "output_dir_name"
            ) -> None:
        cfg = dict(config[task_key])
        self.batch_size: int = cfg["batch_size"]
        self.data_dir = Path(config[engine_key]["data_dir"]).resolve()
        self.max_epochs: int = cfg["max_epochs"]
        self.max_steps: int = cfg.get("max_steps", None)
        self.target_metric: str = cfg["target_metric"]
        self.target_metric_mode: str = cfg["target_metric_mode"]
        self.workers = config[engine_key]["workers"]
        cfg["data_dir"] = self.data_dir
        cfg["workers"] = self.workers
        super().__init__(cfg, identifier_key, outdir_key)


class EngineConfig(BaseConfig):
    def __init__(self, config: dict[str, Any], task_key: str, engine_key: str) -> None:
        cfg = dict(config[engine_key])
        self.accelerator = cfg["accelerator"]
        self.deterministic: bool | Literal["warn"] = cfg["deterministic"]
        self.data_dir = Path(cfg["data_dir"]).resolve()
        self.detect_anomaly: bool = cfg["detect_anomaly"]
        self.devices: int = cfg.get("devices", 1)
        if cfg["early_stopping"] is not None:
            self.early_stopping: int = cfg["early_stopping"]
        else:
            self.early_stopping: int = config[task_key]["max_epochs"]
        self.gradient_clip_alg: str = cfg["gradient_clip_alg"]
        self.gradient_clip_val: float | None = cfg["gradient_clip_val"]
        self.log_extra: bool = cfg.get("log_extra", False)
        self.max_steps: int = config[task_key].get("max_steps", None)
        self.optimize_memory: bool = cfg.get("optimize_memory", False)
        self.output_dir = Path(cfg["output_dir"]).resolve()
        self.precision: str = cfg["precision"]
        resume = cfg.get("resume", False)
        self.resume: Optional[Path] | bool = Path(resume).resolve() if isinstance(resume, str) else resume
        self.run_scheduler: str = cfg["run_scheduler"]
        self.seed: int = cfg["seed"]
        self.seed_mode: str = cfg["seed_mode"]
        self.sbatch_args: dict[str, str] = cfg["sbatch_args"]
        self.silent: bool = cfg.get("silent", False)
        self.test: bool = cfg.get("test", True)
        self.train: bool = cfg.get("train", True)
        self.workers: int = cfg["workers"]
        cfg["max_steps"] = self.max_steps
        cfg["early_stopping"] = self.early_stopping
        super().__init__(cfg)

    def outpath_relevant_engine_keys(self, prefix: str = "") -> list[str]:
        keys = [
            "accelerator",
            "deterministic",
            "detect_anomaly",
            "devices",
            "early_stopping",
            "gradient_clip_alg",
            "gradient_clip_val",
            "optimize_memory",
            "precision",
            "seed"
        ]
        return [f"{prefix}{k}" for k in keys]

    def outpath_irrelevant_engine_keys(self, prefix: str = "") -> list[str]:
        return [f"{prefix}{k}" for k in self.keys() if k not in self.outpath_relevant_engine_keys()]


class EvalConfig(BaseConfig):
    def __init__(self, config: dict[str, Any], eval_key: str, optimizer_key: str, identifier_key: str, ignore_keys = None) -> None:
        cfg = dict(config[eval_key])
        self.experiment_files = AttributeDict(dict(
            best_model = "results_best_model.json",
            last_model = "results_final_model.json",
            config = "config.yaml"
        ))
        self.output_types: list[str] = wrap_list(cfg["output_types"])
        self.output_dir = Path(cfg["output_dir"]).resolve()
        self.experiment_name: str = cfg["experiment_name"]
        self.verbose: bool = cfg.get("verbose", False)
        self.split_groups: bool = cfg.get("split_groups", False)  # TODO: option to split into multiple plots
        self.last_instead_of_best: bool = cfg.get("last_instead_of_best", False)  # TODO: give list of ["last", "best"] (easy: for-loop in lazy_plot)
        column_split_key = cfg.get("column_split_key", None)
        self.column_split_key: Optional[str]  = some(column_split_key, default=f"{optimizer_key}.{identifier_key}")
        self.ignore_keys: list[str] = some(ignore_keys, default=[])
        cfg["ignore_keys"] = self.ignore_keys
        cfg["plot"]["x_axis"] = wrap_list(cfg["plot"]["x_axis"])
        cfg["plot"]["y_axis"] = wrap_list(cfg["plot"]["y_axis"])
        # TODO: columns for multiple plots (ugly: group dataframe by user-specified columns before call to `create_figure`)
        super().__init__(cfg)
