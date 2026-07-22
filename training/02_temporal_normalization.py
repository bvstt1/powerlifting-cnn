import numpy as np
from pathlib import Path
from scipy import interpolate


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "data" / "preprocessed" / "raw_sequences.npz"
OUTPUT_DIR = SCRIPT_DIR / "data" / "normalized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FRAMES = 170


def normalize_temporal(seq, target_frames):
    T, K, C = seq.shape
    if T == target_frames:
        return seq.copy()

    normalized = np.zeros((target_frames, K, C), dtype=np.float32)

    for k in range(K):
        for c in range(C):
            col = seq[:, k, c]
            valid = ~np.isnan(col)
            if valid.sum() == 0:
                normalized[:, k, c] = np.nan
                continue
            if valid.sum() == 1:
                normalized[:, k, c] = col[valid][0]
                continue

            old_idx = np.where(valid)[0]
            old_vals = col[old_idx]
            new_idx = np.linspace(0, T - 1, target_frames)

            f = interpolate.interp1d(
                old_idx, old_vals, kind="linear",
                bounds_error=False, fill_value="extrapolate"
            )
            normalized[:, k, c] = f(new_idx)

    return normalized


def process():
    data = np.load(INPUT_PATH, allow_pickle=True)
    video_ids = data["video_ids"]
    keypoints = data["keypoints"]
    labels = data["labels"]

    print(f"Cargadas {len(video_ids)} secuencias")
    print(f"Normalizando a {TARGET_FRAMES} frames...")

    normalized_kps = []
    for i, kp in enumerate(keypoints):
        norm = normalize_temporal(kp, TARGET_FRAMES)
        normalized_kps.append(norm)

    normalized_kps = np.array(normalized_kps, dtype=np.float32)
    print(f"Shape final: {normalized_kps.shape}")

    out_path = OUTPUT_DIR / "sequences_170.npz"
    np.savez_compressed(
        out_path,
        video_ids=video_ids,
        keypoints=normalized_kps,
        labels=labels,
    )
    print(f"Guardado en: {out_path}")

    nan_after = np.isnan(normalized_kps).sum()
    print(f"NaN restantes: {nan_after}")


if __name__ == "__main__":
    process()
