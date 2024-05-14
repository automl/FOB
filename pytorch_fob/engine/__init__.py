from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent
