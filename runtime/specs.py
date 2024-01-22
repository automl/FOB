import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class SubmissionSpecs():
    """
    Hold information about a workload to be passed to a submission.
    Args:
        `max_epochs`: number of epochs the workload is trained for. This will be changed to `max_steps` later.
    """
    max_epochs: int
    max_steps: Optional[int]

    def __post_init__(self):
        if not self.max_steps:
            raise Exception("Warning: not setting max_steps can lead to bad learning rates!")


@dataclass
class RuntimeSpecs(SubmissionSpecs):
    """
    Hold information about a workload for runtime purposes.
    Args:
        `devices`: Which devices to use for training, see the `lightning.Trainer` argument of the same name.
        `target_metric`: The metric which decides the performance of the model.
        Options are all logged metrics of the model.
        `target_metric_mode`: Whether the target metric should be minimized or maximized. Options are 'min', 'max'.
    """
    devices: list[int] | str | int
    target_metric: str
    target_metric_mode: str

    def export_settings(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "runtime_specs.json", "w", encoding="utf8") as f:
            json.dump(self.__dict__, f, indent=4)


def to_submission_specs(runtime_specs: RuntimeSpecs) -> SubmissionSpecs:
    return SubmissionSpecs(
        max_epochs=runtime_specs.max_epochs,
        max_steps=runtime_specs.max_steps
    )
