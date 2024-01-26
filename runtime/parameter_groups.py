
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional
from torch.nn import Module
from torch.nn.parameter import Parameter


@dataclass
class ParameterGroup():
    named_parameters: Iterator[tuple[str, Parameter]]
    lr_multiplier: float = field(default=1.)
    weight_decay_multiplier: float = field(default=1.)
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_optimizer_dict(
            self,
            lr: Optional[float] = None,
            weight_decay: Optional[float] = None
    ) -> dict[str, Iterator[Parameter] | Any]:
        np = sorted(self.named_parameters)
        d = {
            "params": list(map(lambda x: x[1], np)),
            "names": list(map(lambda x: x[0], np)),
            **self.optimizer_kwargs
        }
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
        return [ParameterGroup(self.model.named_parameters())]

    def grouped_parameters(
            self,
            lr: Optional[float] = None,
            weight_decay: Optional[float] = None
    ) -> list[dict[str, Iterator[Parameter] | Any]]:
        return [pg.to_optimizer_dict(lr, weight_decay) for pg in self.parameter_groups()]


def resolve_parameter_dicts( dict1: dict[str, Any], dict2: dict[str, Any]) -> list[dict[str, Any]]:
    p1, p2 = dict1["params"], dict2["params"]
    n1, n2 = set(dict1["names"]), set(dict2["names"])
    n_to_p1 = dict(zip(dict1["names"], dict1["params"]))
    n_to_p2 = dict(zip(dict2["names"], dict2["params"]))
    assert len(n1) == len(p1)
    assert len(n2) == len(p2)
    kwarg1 = {k: v for k, v in dict1.items() if k not in ["params", "names"]}
    kwarg2 = {k: v for k, v in dict2.items() if k not in ["params", "names"]}
    n1_and_n2 = n1 & n2
    n1_no_n2 = n1 - n2
    n2_no_n1 = n2 - n1
    assert n1_and_n2 | n1_no_n2 | n2_no_n1 == n1 | n2
    outdict1 = {"params": [n_to_p1[n] for n in sorted(n1_no_n2)],
                "names": sorted(n1_no_n2), **kwarg1}
    outdict2 = {"params": [n_to_p2[n] for n in sorted(n2_no_n1)],
                "names": sorted(n2_no_n1), **kwarg2}
    # kwarg2 takes precedence if an arg is present in both dicts:
    outdict12 = {"params": [{**n_to_p1, **n_to_p2}[n] for n in sorted(n1_and_n2)],
                 "names": sorted(n1_and_n2), **kwarg1, **kwarg2}
    return [outdict1, outdict2, outdict12]

def intersect_parameter_dicts(dict1: dict[str, Any], dict2: dict[str, Any]) -> Optional[dict[str, Any]]:
    d = resolve_parameter_dicts(dict1, dict2)[2]
    return d if len(d["params"]) > 0 else None

def merge_parameter_dicts(dict1: dict[str, Any], dict2: dict[str, Any]) -> list[dict[str, Any]]:
    d = resolve_parameter_dicts(dict1, dict2)
    return list(filter(lambda x: len(x["params"]) > 0, d))
