def import_submission(name: str):
    import importlib
    return importlib.import_module(f"submissions.{name}.submission")


def submission_names() -> list[str]:
    from pathlib import Path
    EXCLUDE = ["__pycache__"]
    return [d.name for d in Path(__file__).parent.iterdir() if d.is_dir() and d.name not in EXCLUDE]