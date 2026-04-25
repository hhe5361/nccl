#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Optional


SCRIPT_DIR = Path(__file__).resolve().parent
PHASE2_BASE_PATH = SCRIPT_DIR / "phase2_log_reporter.py"
PHASE3_HELPER_PATH = SCRIPT_DIR.parent / "phase3" / "phase3_log_reporter.py"
SWITCH_LOG_DIR_OVERRIDE: Optional[str] = None


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("phase2_log_reporter_base", PHASE2_BASE_PATH)
switch_helpers = load_module("phase3_switch_helpers_for_phase2", PHASE3_HELPER_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Phase 2 PNG plots and HTML report")
    parser.add_argument("--input", required=True, help="Single experiment root or matrix root")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to <input>/report")
    parser.add_argument("--top-events", type=int, default=12, help="Number of top NCCL events to visualize")
    parser.add_argument("--switch-log-dir", default=None, help="Override switch log directory for all experiments in this report")
    return parser.parse_args()


def resolve_switch_log_dir(experiment_root: Path, env_setup: Optional[dict]) -> Optional[Path]:
    return switch_helpers.resolve_switch_log_dir(experiment_root, env_setup, SWITCH_LOG_DIR_OVERRIDE)


def load_switch_bundle(experiment_root: Path, env_setup: Optional[dict]) -> Optional[dict]:
    return switch_helpers.load_switch_bundle(experiment_root, env_setup, SWITCH_LOG_DIR_OVERRIDE)


def save_switch_overlay_plot(path: Path, experiment_root: Path, env_setup: Optional[dict], mode_data):
    return switch_helpers.save_switch_overlay_plot(path, experiment_root, env_setup, mode_data, SWITCH_LOG_DIR_OVERRIDE)


base.parse_args = parse_args
base.resolve_switch_log_dir = resolve_switch_log_dir
base.load_switch_bundle = load_switch_bundle
base.save_switch_overlay_plot = save_switch_overlay_plot


def main() -> None:
    global SWITCH_LOG_DIR_OVERRIDE
    args = parse_args()
    SWITCH_LOG_DIR_OVERRIDE = args.switch_log_dir
    base.main()


if __name__ == "__main__":
    main()
