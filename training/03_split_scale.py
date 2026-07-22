import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "data" / "normalized" / "sequences_170.npz"
OUTPUT_DIR = SCRIPT_DIR / "data" / "split"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.15
VAL_SIZE = 0.15


def split_and_scale():
    data = np.load(INPUT_PATH)
    X = data["keypoints"]
    y = data["labels"]
    video_ids = data["video_ids"]

    N, T, K, C = X.shape
    print(f"Total: {N} secuencias, {T} frames, {K} keypoints, {C} features")
    print(f"Distribución: válidos={sum(y==1)}, inválidos={sum(y==0)}")

    X_train, X_temp, y_train, y_temp, ids_train, ids_temp = train_test_split(
        X, y, video_ids, test_size=(VAL_SIZE + TEST_SIZE),
        stratify=y, random_state=42
    )

    val_frac = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    X_val, X_test, y_val, y_test, ids_val, ids_test = train_test_split(
        X_temp, y_temp, ids_temp, test_size=0.5,
        stratify=y_temp, random_state=42
    )

    print(f"\nTrain: {len(X_train)} (válidos={sum(y_train==1)}, inválidos={sum(y_train==0)})")
    print(f"Val:   {len(X_val)} (válidos={sum(y_val==1)}, inválidos={sum(y_val==0)})")
    print(f"Test:  {len(X_test)} (válidos={sum(y_test==1)}, inválidos={sum(y_test==0)})")

    N_train = X_train.shape[0]
    X_train_flat = X_train.reshape(N_train, -1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)

    N_val = X_val.shape[0]
    X_val_scaled = scaler.transform(X_val.reshape(N_val, -1)).reshape(X_val.shape)

    N_test = X_test.shape[0]
    X_test_scaled = scaler.transform(X_test.reshape(N_test, -1)).reshape(X_test.shape)

    out_path = OUTPUT_DIR / "split_data.npz"
    np.savez_compressed(
        out_path,
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_val_scaled,
        y_val=y_val,
        X_test=X_test_scaled,
        y_test=y_test,
        ids_train=ids_train,
        ids_val=ids_val,
        ids_test=ids_test,
    )
    print(f"\nGuardado en: {out_path}")

    scaler_path = OUTPUT_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler guardado en: {scaler_path}")

    print(f"\nX_train shape: {X_train_scaled.shape}")
    print(f"X_val shape:   {X_val_scaled.shape}")
    print(f"X_test shape:  {X_test_scaled.shape}")


if __name__ == "__main__":
    split_and_scale()
