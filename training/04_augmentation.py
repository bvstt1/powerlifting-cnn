import numpy as np
from pathlib import Path

from train_utils import augment_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "data" / "split" / "split_data.npz"
OUTPUT_DIR = SCRIPT_DIR / "data" / "augmented"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUG_FACTOR = 3
RANDOM_SEED = 42


def process():
    data = np.load(INPUT_PATH)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print(f"Train original: {X_train.shape[0]}")

    rng = np.random.RandomState(RANDOM_SEED)
    X_train_aug, y_train_aug = augment_dataset(X_train, y_train, rng, AUG_FACTOR)

    validos = sum(y_train_aug == 1)
    invalidos = sum(y_train_aug == 0)
    print(f"Train aumentado: {len(X_train_aug)} (válidos={validos}, inválidos={invalidos})")

    out_path = OUTPUT_DIR / "augmented_data.npz"
    np.savez_compressed(
        out_path,
        X_train=X_train_aug,
        y_train=y_train_aug,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
    print(f"Guardado en: {out_path}")


if __name__ == "__main__":
    process()
