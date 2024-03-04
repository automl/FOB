from pathlib import Path
from typing import Any, Optional


class BaseConfig():
    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    def __getitem__(self, key: str) -> Any:
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
    pass


class WorkloadConfig(NamedConfig):
    def __init__(
            self,
            config: dict[str, Any],
            identifier_key: str = "name",
            outdir_key: str = "output_dir_name"
        ) -> None:
        super().__init__(config, identifier_key, outdir_key)
        self.batch_size: int = config["batch_size"]
        self.max_epochs: int = config["max_epochs"]
        self.max_steps: int = config["max_steps"]
        self.model: str | dict[str, Any] = config["model"]
        self.target_metric: str = config["target_metric"]
        self.target_metric_mode: str = config["target_metric_mode"]


class RuntimeConfig(BaseConfig):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.deterministic: bool = config.get("deterministic", True)
        self.devices: int = config["devices"]
        self.data_dir = Path(config["data_dir"]).resolve()
        self.log_extra: bool = config.get("log_extra", False)
        self.optimize_memory: bool = config.get("optimize_memory", False)
        self.output_dir = Path(config["output_dir"]).resolve()
        maybe_resume = config.get("resume", None)
        self.resume: Optional[Path] = Path(maybe_resume).resolve() if maybe_resume is not None else None
        self.seed: int = config["seed"]
        self.seed_mode: str = config["seed_mode"]
        self.silent: bool = config.get("silent", False)
        self.test_only: bool = config.get("test_only", False)
        self.workers: int = config["workers"]
