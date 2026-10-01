import numpy as np
from scipy import interpolate

from training_config import build_sample_metadata, parse_config


def normalize_temporal(sequence, target_frames):
    frames, keypoints, channels = sequence.shape

    if frames == target_frames:
        return sequence.copy()

    normalized = np.zeros(
        (target_frames, keypoints, channels),
        dtype=np.float32,
    )
    new_index = np.linspace(0, frames - 1, target_frames)

    for keypoint in range(keypoints):
        for channel in range(channels):
            column = sequence[:, keypoint, channel]
            valid = ~np.isnan(column)

            if valid.sum() == 0:
                normalized[:, keypoint, channel] = np.nan
                continue

            if valid.sum() == 1:
                normalized[:, keypoint, channel] = column[valid][0]
                continue

            old_index = np.where(valid)[0]
            old_values = column[old_index]
            function = interpolate.interp1d(
                old_index,
                old_values,
                kind="linear",
                bounds_error=False,
                fill_value=(old_values[0], old_values[-1]),
            )
            normalized[:, keypoint, channel] = function(new_index)

    return normalized


def process(config):
    config["normalized_dir"].mkdir(parents=True, exist_ok=True)
    data = np.load(config["raw_path"], allow_pickle=True)
    video_ids = data["video_ids"]
    keypoints = data["keypoints"]
    labels = data["labels"]
    if "group_ids" in data.files and "source_views" in data.files:
        group_ids = data["group_ids"]
        source_views = data["source_views"]
    else:
        group_ids, source_views = build_sample_metadata(video_ids, config)

    print(f"Perfil: {config['profile']} ({config['display_name']})")
    print(f"Cargadas {len(video_ids)} secuencias")
    print(f"Normalizando a {config['target_frames']} frames...")

    normalized_keypoints = np.asarray([
        normalize_temporal(sequence, config["target_frames"])
        for sequence in keypoints
    ], dtype=np.float32)

    if np.isnan(normalized_keypoints).any():
        raise ValueError("Quedaron NaN despues de la normalizacion temporal")

    np.savez_compressed(
        config["normalized_path"],
        video_ids=video_ids,
        group_ids=group_ids,
        source_views=source_views,
        keypoints=normalized_keypoints,
        labels=labels,
    )

    print(f"Shape final: {normalized_keypoints.shape}")
    print(f"NaN restantes: {np.isnan(normalized_keypoints).sum()}")
    print(f"Guardado en: {config['normalized_path']}")


if __name__ == "__main__":
    process(parse_config("Normalizacion temporal de secuencias"))
