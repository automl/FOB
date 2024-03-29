
from dataclasses import dataclass, field
from typing import Any, Optional
from torch import nn
from torch.nn import Module
from torch.nn.parameter import Parameter
from .utils import some


@dataclass
class ParameterGroup():
    named_parameters: dict[str, Parameter]
    lr_multiplier: Optional[float] = field(default=None)
    weight_decay_multiplier: Optional[float] = field(default=None)
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)

    def __and__(self, other) -> "ParameterGroup":
        assert isinstance(other, ParameterGroup)
        n1 = set(self.named_parameters.keys())
        n2 = set(other.named_parameters.keys())
        all_params = self.named_parameters | other.named_parameters
        n12 = n1 & n2
        new_params = {n: all_params[n] for n in n12}
        return ParameterGroup(
            named_parameters=new_params,
            lr_multiplier=some(other.lr_multiplier, default=self.lr_multiplier),
            weight_decay_multiplier=some(other.weight_decay_multiplier, default=self.weight_decay_multiplier),
            optimizer_kwargs=self.optimizer_kwargs | other.optimizer_kwargs
        )

    def __len__(self) -> int:
        return len(self.named_parameters)

    def __bool__(self) -> bool:
        return not self.empty()

    def empty(self) -> bool:
        return len(self.named_parameters) == 0

    def to_optimizer_dict(
            self,
            lr: Optional[float] = None,
            weight_decay: Optional[float] = None
    ) -> dict[str, list[Parameter] | Any]:
        names = sorted(self.named_parameters)
        d = {
            "params": [self.named_parameters[n] for n in names],
            "names": names,
            **self.optimizer_kwargs
        }
        if lr is not None:
            d["lr"] = self.lr_multiplier * lr if self.lr_multiplier is not None else lr
        if weight_decay is not None:
            d["weight_decay"] = self.weight_decay_multiplier * weight_decay \
            if self.weight_decay_multiplier is not None else weight_decay
        return d


class GroupedModel(Module):
    """
    Wrapper around a nn.Module to allow specifying different optimizer settings for different parameters.
    To use this feature for your task, inherit from this class and override the `parameter_groups` method.
    Then simply wrap your model before passing it to the `__init__` method of the `TaskModel` superclass.
    """
    def __init__(self, model: Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def parameter_groups(self) -> list[ParameterGroup]:
        return wd_group_named_parameters(self.model)

    def grouped_parameters(
            self,
            lr: Optional[float] = None,
            weight_decay: Optional[float] = None
    ) -> list[dict[str, list[Parameter] | Any]]:
        return [pg.to_optimizer_dict(lr, weight_decay) for pg in self.parameter_groups()]


def merge_parameter_splits(split1: list[ParameterGroup], split2: list[ParameterGroup]) -> list[ParameterGroup]:
    groups = []
    for pg1 in split1:
        for pg2 in split2:
            pg12 = pg1 & pg2
            if not pg12.empty():
                groups.append(pg12)
    return groups


def wd_group_named_parameters(model: Module):
    apply_decay = set()
    apply_no_decay = set()
    special = set()
    whitelist_weight_modules = (nn.Linear, nn.Conv2d)
    ignore_modules = (nn.Sequential, )
    blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                nn.Embedding,
                                nn.LazyBatchNorm1d, nn.LazyBatchNorm2d, nn.LazyBatchNorm3d,
                                nn.GroupNorm, nn.SyncBatchNorm,
                                nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                                nn.LayerNorm, nn.LocalResponseNorm)

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn  # full param name
            if not p.requires_grad or fpn not in param_dict:
                continue  # frozen weights
            if isinstance(m, ignore_modules):
                continue  # parameters of sequential are added from their own modules
            if hasattr(p, '_optim'):
                special.add(fpn)
            elif pn.endswith('bias'):
                apply_no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                apply_decay.add(fpn)
            elif isinstance(m, blacklist_weight_modules):
                apply_no_decay.add(fpn)
            # else:  # for debug purposes
            #     print("wd_group_named_parameters: Not using any rule for ", fpn, " in ", type(m))

    apply_decay |= (param_dict.keys() - apply_no_decay - special)

    # validate that we considered every parameter
    inter_params = apply_decay & apply_no_decay
    union_params = apply_decay | apply_no_decay
    assert len(inter_params) == 0, f"Parameters {str(inter_params)} made it into both apply_decay/apply_no_decay sets!"
    assert len(
        param_dict.keys() - special - union_params) == 0, \
            f"parameters {str(param_dict.keys() - union_params)} \
                were not separated into either apply_decay/apply_no_decay set!"

    if not apply_no_decay:
        param_groups = [ParameterGroup(
            named_parameters=dict(zip(sorted(union_params), (param_dict[pn] for pn in sorted(union_params))))
        )]
    else:
        param_groups = [
            ParameterGroup(
                named_parameters=dict(zip(sorted(apply_no_decay), (param_dict[pn] for pn in sorted(apply_no_decay)))),
                weight_decay_multiplier=0.
            ),
            ParameterGroup(
                named_parameters=dict(zip(sorted(apply_decay), (param_dict[pn] for pn in sorted(apply_decay))))
            ),
        ]

    return param_groups


def resolve_parameter_dicts(dict1: dict[str, Any], dict2: dict[str, Any]) -> list[dict[str, Any]]:
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
