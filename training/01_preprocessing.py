import numpy as np
import pandas as pd

from training_config import (
    MANUAL_LABEL_AUDIT_PATH,
    build_sample_metadata,
    parse_config,
)


MAX_CONSECUTIVE_NAN = 5


def load_keypoints(video_id, config):
    path = config["keypoints_dir"] / f"{video_id}.npy"

    if not path.exists():
        return None

    return np.load(str(path)).astype(np.float32)


def validate_and_clean(kp, expected_keypoints):
    if kp is None or kp.ndim != 3:
        return None

    if kp.shape[1:] != (expected_keypoints, 3):
        return None

    if kp.shape[0] < 10 or np.isnan(kp).all() or np.isinf(kp).any():
        return None

    return kp


def interpolate_small_gaps(kp, max_gap=MAX_CONSECUTIVE_NAN):
    frames, keypoints, channels = kp.shape
    cleaned = kp.copy()

    for keypoint in range(keypoints):
        for channel in range(channels):
            column = cleaned[:, keypoint, channel]
            nan_mask = np.isnan(column)

            if nan_mask.sum() == 0 or nan_mask.sum() == frames:
                continue

            index = 0

            while index < frames:
                if not nan_mask[index]:
                    index += 1
                    continue

                gap_start = index

                while index < frames and nan_mask[index]:
                    index += 1

                gap_end = index
                gap_length = gap_end - gap_start

                if gap_length > max_gap:
                    continue

                has_before = gap_start > 0
                has_after = gap_end < frames

                if has_before and has_after:
                    before = column[gap_start - 1]
                    after = column[gap_end]

                    for gap_index in range(gap_length):
                        alpha = (gap_index + 1) / (gap_length + 1)
                        column[gap_start + gap_index] = (
                            before + (after - before) * alpha
                        )
                elif has_after:
                    column[gap_start:gap_end] = column[gap_end]
                elif has_before:
                    column[gap_start:gap_end] = column[gap_start - 1]

    return cleaned


def normalize_spatial_front(kp):
    frames = kp.shape[0]
    normalized = kp.copy()
    left_hip_idx = 12
    right_hip_idx = 13
    left_shoulder_idx = 0

    for frame in range(frames):
        left_hip = normalized[frame, left_hip_idx, :2]
        right_hip = normalized[frame, right_hip_idx, :2]

        if np.isnan(left_hip).any() or np.isnan(right_hip).any():
            continue

        mid_hip = (left_hip + right_hip) / 2.0
        shoulder = normalized[frame, left_shoulder_idx, :2].copy()

        if np.isnan(shoulder).any():
            continue

        torso_height = abs(shoulder[1] - mid_hip[1])
        normalized[frame, :, :2] -= mid_hip

        if torso_height > 1e-6:
            normalized[frame, :, :2] /= torso_height

    return normalized


def normalize_spatial_side(kp, source_view):
    normalized = kp.copy()
    shoulder = normalized[:, 0, :2]
    hip = normalized[:, 1, :2]
    valid = (~np.isnan(shoulder).any(axis=1)) & (~np.isnan(hip).any(axis=1))
    valid_indices = np.where(valid)[0]

    if len(valid_indices) == 0:
        return normalized

    reference_indices = valid_indices[:min(10, len(valid_indices))]
    hip_reference = np.median(hip[reference_indices], axis=0)
    torso_lengths = np.linalg.norm(shoulder[valid] - hip[valid], axis=1)
    valid_lengths = torso_lengths[torso_lengths > 1e-6]
    torso_scale = float(np.median(valid_lengths)) if len(valid_lengths) else 1.0

    normalized[:, :, :2] -= hip_reference

    if torso_scale > 1e-6:
        normalized[:, :, :2] /= torso_scale

    if source_view == "left":
        normalized[:, :, 0] *= -1

    return normalized


def normalize_spatial(kp, config, source_view):
    if config["normalization"] == "dl_side":
        return normalize_spatial_side(kp, source_view)

    return normalize_spatial_front(kp)


def load_labels(config):
    df = pd.read_csv(config["csv_path"])

    if df["video_id"].duplicated().any():
        duplicated = df.loc[df["video_id"].duplicated(), "video_id"].tolist()
        raise ValueError(f"IDs duplicados en CSV: {duplicated[:10]}")

    if df["label"].isna().any():
        raise ValueError("El CSV contiene etiquetas vacias")

    labels = df["label"].astype(int)

    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError("Las etiquetas deben ser binarias: 0 o 1")

    return dict(zip(df["video_id"].astype(str), labels))


def process_all(config):
    config["preprocessed_dir"].mkdir(parents=True, exist_ok=True)
    label_map = load_labels(config)
    keypoint_paths = sorted(config["keypoints_dir"].glob("*.npy"))
    processed = []
    skipped = []

    all_ids = [path.stem for path in keypoint_paths]
    all_group_ids, all_source_views = build_sample_metadata(all_ids, config)
    source_view_map = dict(zip(all_ids, all_source_views))
    group_id_map = dict(zip(all_ids, all_group_ids))
    excluded_events = set()

    if MANUAL_LABEL_AUDIT_PATH.exists() and config["profile"] in (
        "dl_left", "dl_right", "dl_side"
    ):
        manual_audit = pd.read_csv(MANUAL_LABEL_AUDIT_PATH)
        excluded_events = set(
            manual_audit.loc[
                manual_audit["discarded"].astype(int) == 1,
                "physical_event",
            ].astype(str)
        )

    for path in keypoint_paths:
        video_id = path.stem

        if group_id_map[video_id] in excluded_events:
            skipped.append((video_id, "manual_review_discard"))
            continue

        if video_id not in label_map:
            skipped.append((video_id, "sin etiqueta"))
            continue

        keypoints = load_keypoints(video_id, config)
        keypoints = validate_and_clean(
            keypoints,
            config["expected_keypoints"],
        )

        if keypoints is None:
            skipped.append((video_id, "forma invalida/corrupto/vacio"))
            continue

        keypoints = interpolate_small_gaps(keypoints)
        remaining_nan = np.isnan(keypoints[:, :, :2]).any(axis=(1, 2))

        if remaining_nan.sum() > keypoints.shape[0] * 0.3:
            reason = f"demasiados NaN ({remaining_nan.sum()}/{keypoints.shape[0]})"
            skipped.append((video_id, reason))
            continue

        source_view = source_view_map[video_id]
        keypoints = normalize_spatial(keypoints, config, source_view)

        processed.append({
            "video_id": video_id,
            "keypoints": keypoints,
            "label": label_map[video_id],
        })

    if not processed:
        raise RuntimeError("No se proceso ninguna secuencia")

    processed_ids = np.asarray([item["video_id"] for item in processed])
    group_ids, source_views = build_sample_metadata(processed_ids, config)

    np.savez_compressed(
        config["raw_path"],
        video_ids=processed_ids,
        group_ids=group_ids,
        source_views=source_views,
        keypoints=np.array(
            [item["keypoints"] for item in processed],
            dtype=object,
        ),
        labels=np.asarray([item["label"] for item in processed]),
    )

    print(f"Perfil: {config['profile']} ({config['display_name']})")
    print(f"Procesados: {len(processed)}")
    print(f"Omitidos: {len(skipped)}")

    for video_id, reason in skipped:
        print(f"  - {video_id}: {reason}")

    frame_counts = [item["keypoints"].shape[0] for item in processed]
    labels = np.asarray([item["label"] for item in processed])
    print("\nFrames por secuencia:")
    print(
        f"  min={min(frame_counts)}, max={max(frame_counts)}, "
        f"mean={np.mean(frame_counts):.1f}"
    )
    print(f"  labels: validos={sum(labels == 1)}, invalidos={sum(labels == 0)}")
    print(f"\nGuardado en: {config['raw_path']}")


if __name__ == "__main__":
    process_all(parse_config("Preprocesamiento de secuencias de pose"))
