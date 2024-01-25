
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter


@dataclass
class ParameterGroup():
    parameters: Iterator[Parameter]
    lr_multiplier: float = field(default=1.)
    weight_decay_multiplier: float = field(default=1.)
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_optimizer_dict(
            self,
            lr: Optional[float] = None,
            weight_decay: Optional[float] = None
    ) -> dict[str, Iterator[Parameter] | Any]:
        d = {"params": self.parameters, **self.optimizer_kwargs}
        if lr is not None:
            d["lr"] = self.lr_multiplier * lr
        if weight_decay is not None:
            d["weight_decay"] = self.weight_decay_multiplier * weight_decay
        return d


class GroupedModel(Module):
    """
    Wrapper around a nn.Module to allow specifying different optimizer settings for different parameters.
    To use this feature for your workload, inherit from this class and override the `parameter_groups` method.
    Then simply wrap your model before passing it to the `__init__` method of the `WorkloadModel` superclass.
    """
    def __init__(self, model: Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def parameter_groups(self) -> list[ParameterGroup]:
        return [ParameterGroup(self.model.parameters())]

    def grouped_parameters(
            self,
            lr: Optional[float] = None,
            weight_decay: Optional[float] = None
    ) -> list[dict[str, Iterator[Parameter] | Any]]:
        return [param.to_optimizer_dict(lr, weight_decay) for param in self.parameter_groups()]
