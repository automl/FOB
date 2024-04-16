from typing import Any


def unique(l: list) -> list:
    """Returns deduplicated list"""
    res = []
    for v in l:
        if v not in res:
            res.append(v)
    return res


def grid_search(d: dict[str, Any]) -> list[dict[str, Any]]:
    ret = []
    if isinstance(d, dict):
        if len(d) == 0:
            return [dict()]
        copy = d.copy()
        k, v = copy.popitem()
        configs = unique(grid_search(v))
        rest = grid_search(copy)
        for r in rest:
            for config in configs:
                ret.append(r | {k: config})
    elif isinstance(d, list):
        for v in d:
            configs = grid_search(v)
            for config in configs:
                ret.append(config)
    else:
        ret.append(d)
    return ret
