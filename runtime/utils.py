def some(*args, default):
    """
    returns the first argument that is not None or default.
    """
    if len(args) < 1:
        return default
    first, *rest = args
    if first is not None:
        return first
    return some(*rest, default=default)
