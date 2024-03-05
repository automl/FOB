from pathlib import Path
from typing import Any, Optional


class BaseConfig():
    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def __getattribute__(self, key: str) -> Any:
        try:
            return super().__getattribute__(key)
        except AttributeError:
            pass
        return self._config[key]

    def keys(self):
        return self._config.keys()

    def items(self):
        return self._config.items()


class NamedConfig(BaseConfig):
    def __init__(
            self,
            config: dict[str, Any],
            identifier_key: str = "name",
            outdir_key: str = "output_dir_name"
        ) -> None:
        super().__init__(config)
        self.name = config[identifier_key]
        self.output_dir_name = config[outdir_key]


class SubmissionConfig(NamedConfig):
    def __init__(
            self,
            config: dict[str, Any],
            submission_key: str,
            workload_key: str,
            identifier_key: str = "name",
            outdir_key: str = "output_dir_name"
        ) -> None:
        cfg = config[submission_key]
        self.max_steps = config[workload_key]["max_steps"]
        cfg["max_steps"] = self.max_steps
        super().__init__(cfg, identifier_key, outdir_key)


class WorkloadConfig(NamedConfig):
    def __init__(
            self,
            config: dict[str, Any],
            workload_key: str,
            runtime_key: str,
            identifier_key: str = "name",
            outdir_key: str = "output_dir_name"
        ) -> None:
        cfg = config[workload_key]
        self.batch_size: int = cfg["batch_size"]
        self.data_dir = Path(config[runtime_key]["data_dir"]).resolve()
        self.max_epochs: int = cfg["max_epochs"]
        self.max_steps: int = cfg["max_steps"]
        self.model: str | dict[str, Any] = cfg["model"]
        self.target_metric: str = cfg["target_metric"]
        self.target_metric_mode: str = cfg["target_metric_mode"]
        self.workers = config[runtime_key]["workers"]
        cfg["data_dir"] = self.data_dir
        cfg["workers"] = self.workers
        super().__init__(cfg, identifier_key, outdir_key)


class RuntimeConfig(BaseConfig):
    def __init__(self, config: dict[str, Any], workload_key: str, runtime_key: str) -> None:
        cfg = config[runtime_key]
        self.deterministic: bool = cfg.get("deterministic", True)
        self.devices: int = cfg["devices"]
        self.data_dir = Path(cfg["data_dir"]).resolve()
        self.log_extra: bool = cfg.get("log_extra", False)
        self.max_steps = config[workload_key]["max_steps"]
        self.optimize_memory: bool = cfg.get("optimize_memory", False)
        self.output_dir = Path(cfg["output_dir"]).resolve()
        maybe_resume = cfg.get("resume", None)
        self.resume: Optional[Path] = Path(maybe_resume).resolve() if maybe_resume is not None else None
        self.seed: int = cfg["seed"]
        self.seed_mode: str = cfg["seed_mode"]
        self.silent: bool = cfg.get("silent", False)
        self.test_only: bool = cfg.get("test_only", False)
        self.workers: int = cfg["workers"]
        cfg["max_steps"] = self.max_steps
        super().__init__(cfg)
