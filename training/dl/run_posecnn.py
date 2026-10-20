"""Entrada por etapas para DL; conserva los scripts numerados historicos.

Ejemplo: python -m training.dl.run_posecnn --profile dl_left --stage split
"""

import argparse
import importlib

from training.configs.training_config import get_config


STAGES = {
    "preprocess": ("01_preprocessing", "process_all"),
    "temporal": ("02_temporal_normalization", "process"),
    "split": ("03_split_scale", "split_and_scale"),
    "augment": ("04_augmentation", "process"),
    "train": ("05_train_cnn", "main"),
    "cv": ("06_cross_validation", "run_cross_validation"),
    "report": ("07_report", "generate_report"),
}


def main():
    parser = argparse.ArgumentParser(description="Pipeline PoseCNN Deadlift")
    parser.add_argument("--profile", choices=("dl_front", "dl_left", "dl_right", "dl_side"), required=True)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    args = parser.parse_args()
    modulo, funcion = STAGES[args.stage]
    getattr(importlib.import_module(f"training.dl.{modulo}"), funcion)(get_config(args.profile))


if __name__ == "__main__":
    main()
