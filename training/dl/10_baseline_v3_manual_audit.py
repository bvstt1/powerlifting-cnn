"""Wrapper de entrenamiento Baseline V3 post auditoria manual."""

import argparse
import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))
spec = importlib.util.spec_from_file_location(
    "baseline_runner", SCRIPT_DIR / "08_baseline_v2.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

runner.EXPERIMENT = "baseline_v3_manual_audit"
runner.HISTORICAL = {
    "dl_left": {"f1": 0.7667, "roc_auc": 0.8820, "note": "Baseline V2"},
    "dl_right": {"f1": 0.8308, "roc_auc": 0.8942, "note": "Baseline V2"},
    "dl_side": {"f1": 0.8120, "roc_auc": 0.8682, "note": "Baseline V2"},
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline V3 manual audit")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--profile", choices=("dl_left", "dl_right", "dl_side"))
    group.add_argument("--paired", action="store_true")
    args = parser.parse_args()

    if args.paired:
        runner.paired_evaluation()
    else:
        runner.train_profile(args.profile)
