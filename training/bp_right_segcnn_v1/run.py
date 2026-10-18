"""Entradas reproducibles RIGHT: python -m training.bp_right_segcnn_v1.run --phase train.

El código común versionado está en training/bp_side_segcnn_v1/ y sus hashes
quedan registrados en source_manifest.json. `prepare` y `all` solo leen
secuencias Train/Validation; `train` reanuda verificando checkpoints.
"""
import argparse
from training.bp_side_segcnn_v1.train import prepare_view, train_view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "train", "all"), required=True)
    phase = parser.parse_args().phase
    if phase in ("prepare", "all"):
        prepare_view("right")
    if phase in ("train", "all"):
        train_view("right")


if __name__ == "__main__":
    main()
