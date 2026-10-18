"""Wrapper 5-Fold CV de Baseline V3 post auditoria manual."""

import argparse
import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))
spec = importlib.util.spec_from_file_location(
    "cv_runner", SCRIPT_DIR / "09_baseline_v2_cv.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

runner.EXPERIMENT = "baseline_v3_manual_audit_cv"
runner.COMMON_FOLDS_PATH = (
    SCRIPT_DIR.parent / "data" / "dl" / "cv_event_folds_v3.csv"
)
runner.COMMON_RESULTS_DIR = (
    SCRIPT_DIR.parent / "results" / "dl" / "baseline_v3_manual_audit_cv"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline V3 manual audit CV")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--profile", choices=runner.PROFILES)
    group.add_argument("--paired", action="store_true")
    args = parser.parse_args()

    if args.prepare:
        runner.prepare_common_folds()
    elif args.profile:
        runner.train_profile(args.profile)
    else:
        runner.paired_cv()
