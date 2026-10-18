import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.common.train_utils import (
    augment_dataset,
    create_scaler,
    scale_data,
    verify_arrays_finite,
    verify_split_shapes,
)
from training.configs.training_config import parse_config


AUG_FACTOR = 3

# Orden real (fase 1, revision 2): el StandardScaler se ajusta
# EXCLUSIVAMENTE con el train ORIGINAL (muestras reales, sin aumentar) para
# que sus estadisticas no dependan del tipo/cantidad de augmentation y las
# ablaciones futuras compartan el mismo sistema de normalizacion. Despues:
# train original -> augmentation -> transform; val/test -> transform.


def process(config):
    config["augmented_dir"].mkdir(parents=True, exist_ok=True)
    data = np.load(config["split_path"])
    X_train = np.asarray(data["X_train"], dtype=np.float32)
    y_train = np.asarray(data["y_train"])
    X_val = np.asarray(data["X_val"], dtype=np.float32)
    y_val = np.asarray(data["y_val"])
    X_test = np.asarray(data["X_test"], dtype=np.float32)
    y_test = np.asarray(data["y_test"])

    n_train_before = len(X_train)
    n_val_before = len(X_val)
    n_test_before = len(X_test)

    print(f"Perfil: {config['profile']} ({config['display_name']})")
    print(f"Train original (sin escalar): {X_train.shape[0]}")
    print(f"Aumentaciones habilitadas: {', '.join(config['augmentation'])}")

    verify_arrays_finite(
        X_train, X_val, X_test,
        names=["X_train_split", "X_val_split", "X_test_split"],
    )

    # 1. Fit del scaler EXCLUSIVAMENTE con train original (muestras reales).
    #    Val/test nunca participan en fit; el train aumentado tampoco.
    scaler = create_scaler(X_train)

    assert int(scaler.n_samples_seen_) == n_train_before, (
        f"El scaler vio {scaler.n_samples_seen_} muestras, "
        f"esperado train original={n_train_before}"
    )

    print(
        f"Scaler ajustado solo en train original: {n_train_before} muestras "
        f"(n_features={len(scaler.mean_)}, "
        f"n_samples_seen={scaler.n_samples_seen_})"
    )

    # 2. Aumento SOLO en train y en espacio sin escalar.
    rng = np.random.RandomState(config["random_seed"])
    X_train_aug, y_train_aug = augment_dataset(
        X_train,
        y_train,
        rng,
        AUG_FACTOR,
        transforms=config["augmentation"],
    )
    X_train_aug = np.asarray(X_train_aug, dtype=np.float32)
    y_train_aug = np.asarray(y_train_aug)

    assert len(X_train_aug) == n_train_before * (AUG_FACTOR + 1), (
        f"Train aumentado={len(X_train_aug)}, "
        f"esperado={n_train_before * (AUG_FACTOR + 1)}"
    )
    assert len(X_val) == n_val_before and len(X_test) == n_test_before, (
        "Val/test no deben aumentar: solo train se aumenta"
    )

    print(
        f"Train aumentado (sin escalar): {len(X_train_aug)} "
        f"(validos={sum(y_train_aug == 1)}, "
        f"invalidos={sum(y_train_aug == 0)})"
    )

    verify_arrays_finite(X_train_aug, names=["X_train_aug_unscaled"])

    # 3. Transform con el scaler ya ajustado (train aumentado solo pasa por
    #    transform; idem val/test).
    X_train_scaled = scale_data(scaler, X_train_aug).astype(np.float32)
    X_val_scaled = scale_data(scaler, X_val).astype(np.float32)
    X_test_scaled = scale_data(scaler, X_test).astype(np.float32)

    verify_arrays_finite(
        X_train_scaled, X_val_scaled, X_test_scaled,
        names=["X_train_scaled", "X_val_scaled", "X_test_scaled"],
    )
    verify_split_shapes(
        X_train_scaled, X_val_scaled, X_test_scaled,
        config["target_frames"], config["expected_keypoints"],
    )

    # Nota: el train transformado YA NO tiene media exactamente 0/std 1
    # porque incluye muestras aumentadas (el scaler se ajusto sin ellas).
    # Se reporta de forma informativa, sin assertion estricta.
    train_flat = X_train_scaled.reshape(len(X_train_scaled), -1)
    print(
        f"Train escalado (incluye aumento): media={float(train_flat.mean()):.6f}, "
        f"std={float(train_flat.std()):.6f}"
    )

    np.savez_compressed(
        config["augmented_path"],
        X_train=X_train_scaled,
        y_train=y_train_aug,
        X_val=X_val_scaled,
        y_val=y_val,
        X_test=X_test_scaled,
        y_test=y_test,
    )

    with open(config["scaler_path"], "wb") as file:
        pickle.dump(scaler, file)

    print(f"Scaler (fit solo en train original) guardado en: {config['scaler_path']}")
    print(f"Guardado en: {config['augmented_path']}")
    print(f"X_train shape: {X_train_scaled.shape}")
    print(f"X_val shape:   {X_val_scaled.shape}")
    print(f"X_test shape:  {X_test_scaled.shape}")
    print("Checks augment+scale: OK "
          "(scaler fit solo en train original, aug solo en train, "
          "train aumentado solo transform, sin NaN/Inf, shapes correctas)")


if __name__ == "__main__":
    process(parse_config("Aumentacion del conjunto de entrenamiento"))
