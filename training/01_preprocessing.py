import numpy as np
import pandas as pd
from pathlib import Path
from scipy import interpolate


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

KEYPOINTS_DIR = PROJECT_DIR / "keypoints" / "dl" / "front"
CSV_PATH = PROJECT_DIR / "etiquetado" / "dl" / "etiquetado_dl_front_fixed.csv"
OUTPUT_DIR = SCRIPT_DIR / "data" / "preprocessed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_KEYPOINTS = 25

# Umbral de visibilidad para considerar un keypoint como detectado
VISIBILITY_THRESHOLD = 0.5
# Maximo de frames consecutivos a interpolar
MAX_CONSECUTIVE_NAN = 5


def load_keypoints(video_id):
    path = KEYPOINTS_DIR / f"{video_id}.npy"
    if not path.exists():
        return None
    return np.load(str(path)).astype(np.float32)


def validate_and_clean(kp):
    if kp is None:
        return None
    if np.isnan(kp).all():
        return None
    if kp.shape[0] < 10:
        return None
    return kp


def interpolate_small_gaps(kp, max_gap=MAX_CONSECUTIVE_NAN):
    T, K, C = kp.shape
    cleaned = kp.copy()
    for k in range(K):
        for c in range(C):
            col = cleaned[:, k, c]
            nan_mask = np.isnan(col)
            if nan_mask.sum() == 0:
                continue
            if nan_mask.sum() == T:
                continue
            valid = np.where(~nan_mask)[0]
            gap_start = None
            for t in range(T):
                if nan_mask[t]:
                    if gap_start is None:
                        gap_start = t
                else:
                    if gap_start is not None:
                        gap_len = t - gap_start
                        if gap_len <= max_gap:
                            before = col[gap_start - 1]
                            after = col[t]
                            for g in range(gap_len):
                                alpha = (g + 1) / (gap_len + 1)
                                col[gap_start + g] = before + (after - before) * alpha
                        gap_start = None
            if gap_start is not None and gap_start <= T - 1:
                gap_len = T - gap_start
                if gap_len <= max_gap and gap_start > 0:
                    before = col[gap_start - 1]
                    for g in range(gap_len):
                        col[gap_start + g] = before
    return cleaned


def normalize_spatial(kp):
    T, K, C = kp.shape
    normalized = kp.copy()

    left_hip_idx = 12
    right_hip_idx = 13
    left_shoulder_idx = 0

    for t in range(T):
        lh = normalized[t, left_hip_idx, :2]
        rh = normalized[t, right_hip_idx, :2]
        if np.isnan(lh).any() or np.isnan(rh).any():
            continue
        mid_hip = (lh + rh) / 2.0
        normalized[t, :, :2] -= mid_hip

        left_shoulder = normalized[t, left_shoulder_idx, :2]
        if np.isnan(left_shoulder).any():
            continue
        torso_height = abs(left_shoulder[1] - mid_hip[1])
        if torso_height > 1e-6:
            normalized[t, :, :2] /= torso_height

    return normalized


def load_labels():
    df = pd.read_csv(CSV_PATH)
    label_map = {}
    for _, row in df.iterrows():
        video_id = row["video_id"]
        label = int(row["label"])
        label_map[video_id] = label
    return label_map


def process_all():
    label_map = load_labels()
    all_video_ids = sorted(KEYPOINTS_DIR.glob("*.npy"))
    processed = []
    skipped = []

    for path in all_video_ids:
        video_id = path.stem

        if video_id not in label_map:
            skipped.append((video_id, "sin etiqueta"))
            continue

        kp = load_keypoints(video_id)
        kp = validate_and_clean(kp)
        if kp is None:
            skipped.append((video_id, "corrupto/vacio"))
            continue

        kp = interpolate_small_gaps(kp)

        remaining_nan = np.isnan(kp[:, :, :2]).any(axis=(1, 2))
        if remaining_nan.sum() > kp.shape[0] * 0.3:
            skipped.append((video_id, f"demasiados NaN ({remaining_nan.sum()}/{kp.shape[0]})"))
            continue

        kp = normalize_spatial(kp)

        label = label_map[video_id]
        out = {"video_id": video_id, "keypoints": kp, "label": label}
        processed.append(out)

    out_path = OUTPUT_DIR / "raw_sequences.npz"
    np.savez_compressed(
        out_path,
        video_ids=[p["video_id"] for p in processed],
        keypoints=np.array([p["keypoints"] for p in processed], dtype=object),
        labels=np.array([p["label"] for p in processed]),
    )

    print(f"Procesados: {len(processed)}")
    print(f"Omitidos: {len(skipped)}")
    for vid, reason in skipped:
        print(f"  - {vid}: {reason}")

    T_values = [p["keypoints"].shape[0] for p in processed]
    print(f"\nFrames por secuencia:")
    print(f"  min={min(T_values)}, max={max(T_values)}, mean={np.mean(T_values):.1f}")
    print(f"  labels: validos={sum(p['label']==1 for p in processed)}, invalidos={sum(p['label']==0 for p in processed)}")
    print(f"\nGuardado en: {out_path}")


if __name__ == "__main__":
    process_all()
