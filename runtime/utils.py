from pathlib import Path
import json
import signal
import torch


def write_results(results, filepath: Path):
    with open(filepath, "w", encoding="utf8") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results into {filepath}.")


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


def trainer_strategy(devices: int | list[int] | str) -> str:
    if isinstance(devices, str):
        return "auto"
    ndevices = devices if isinstance(devices, int) else len(devices)
    return "ddp" if ndevices > 1 else "auto"


def gpu_suited_for_compile():
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        return device_cap in ((7, 0), (8, 0), (9, 0))


def begin_timeout(delay=10, show_threads=False):
    if show_threads:
        import sys
        import traceback
        import threading
        thread_names = {t.ident: t.name for t in threading.enumerate()}
        for thread_id, frame in sys._current_frames().items():
            print("Thread %s:" % thread_names.get(thread_id, thread_id))
            traceback.print_stack(frame)
            print()
    signal.alarm(delay)  # Timeout after 10 seconds


def path_to_str_inside_dict(d: dict) -> dict:
    return {k: (str(v) if isinstance(v, Path) else
                (path_to_str_inside_dict(v) if isinstance(v, dict) else v))
            for k, v in d.items()}
