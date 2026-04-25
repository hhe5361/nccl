#!/usr/bin/env python3
import importlib.util
from pathlib import Path
from types import ModuleType


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / "phase3_log_reporter.py"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("phase3_log_reporter_base", BASE_PATH)


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
