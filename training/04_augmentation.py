import numpy as np
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "data" / "split" / "split_data.npz"
OUTPUT_DIR = SCRIPT_DIR / "data" / "augmented"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUG_FACTOR = 3
RANDOM_SEED = 42
rng = np.random.RandomState(RANDOM_SEED)


def add_translation(seq, max_shift=0.05):
    shift_x = rng.uniform(-max_shift, max_shift)
    shift_y = rng.uniform(-max_shift, max_shift)
    aug = seq.copy()
    aug[..., 0] += shift_x
    aug[..., 1] += shift_y
    return aug


def add_scaling(seq, scale_range=(0.9, 1.1)):
    scale = rng.uniform(*scale_range)
    aug = seq.copy()
    aug[..., :2] *= scale
    return aug


def add_gaussian_noise(seq, noise_std=0.01):
    noise = rng.normal(0, noise_std, size=seq[..., :2].shape).astype(np.float32)
    aug = seq.copy()
    aug[..., :2] += noise
    return aug


def add_rotation(seq, max_angle=10):
    angle = rng.uniform(-max_angle, max_angle)
    angle_rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    aug = seq.copy()
    x = aug[..., 0]
    y = aug[..., 1]
    aug[..., 0] = x * cos_a - y * sin_a
    aug[..., 1] = x * sin_a + y * cos_a
    return aug


def add_temporal_shift(seq, max_shift=3):
    T = seq.shape[0]
    shift = rng.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return seq.copy()
    aug = np.roll(seq, shift, axis=0)
    if shift > 0:
        aug[:shift] = seq[0]
    else:
        aug[shift:] = seq[-1]
    return aug


def augment_sequence(seq, label):
    aug_list = [(seq, label)]

    for _ in range(AUG_FACTOR):
        aug = seq.copy()
        if rng.rand() > 0.5:
            aug = add_translation(aug)
        if rng.rand() > 0.5:
            aug = add_scaling(aug)
        if rng.rand() > 0.5:
            aug = add_gaussian_noise(aug)
        if rng.rand() > 0.5:
            aug = add_rotation(aug)
        if rng.rand() > 0.5:
            aug = add_temporal_shift(aug)
        aug_list.append((aug, label))

    return aug_list


def process():
    data = np.load(INPUT_PATH)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print(f"Train original: {X_train.shape[0]}")

    all_X = []
    all_y = []
    for i in range(len(X_train)):
        augmented = augment_sequence(X_train[i], y_train[i])
        for aug_kp, aug_label in augmented:
            all_X.append(aug_kp)
            all_y.append(aug_label)

    X_train_aug = np.array(all_X, dtype=np.float32)
    y_train_aug = np.array(all_y)

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
