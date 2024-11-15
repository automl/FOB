"""Copied from https://github.com/automl/CPR/blob/main/pytorch_cpr/group_parameter.py and adjusted to include parameter names like in previous version to maintain compatibility."""

from torch import nn


def group_parameters_for_cpr_optimizer(model, regularize_embedding=False):
    decay = set()
    no_decay = set()
    special = set()
    whitelist_weight_modules = (nn.Linear,)

    blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                nn.LazyBatchNorm1d, nn.LazyBatchNorm2d, nn.LazyBatchNorm3d,
                                nn.GroupNorm, nn.SyncBatchNorm,
                                nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                                nn.LayerNorm, nn.LocalResponseNorm)

    if not regularize_embedding:
        blacklist_weight_modules += (nn.Embedding,)

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn
            if not p.requires_grad or fpn not in param_dict:
                continue
            if hasattr(p, '_optim'):
                special.add(fpn)
            elif getattr(p, '_no_weight_decay', False):
                no_decay.add(fpn)
            elif pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)

    decay |= (param_dict.keys() - no_decay - special)
    # validate that we considered every parameter
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"Parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(
        param_dict.keys() - special - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)}  were not separated into either decay/no_decay set!"

    if not no_decay:
        param_groups = [{"params": [param_dict[pn] for pn in sorted(list(no_decay | decay))],
                         "regularize": True}]
    else:
        param_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "names": sorted(list(decay)), "regularize": True},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "names": sorted(list(no_decay)), "regularize": False},
        ]

    hps = [dict(s) for s in set(frozenset(param_dict[pn]._optim.items()) for pn in special)]
    for hp in hps:
        params = [param_dict[pn] for pn in sorted(list(special)) if param_dict[pn]._optim == hp]
        param_groups.append({"params": params, **hp})

    return param_groups
