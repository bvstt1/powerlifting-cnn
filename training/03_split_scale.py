import numpy as np
from sklearn.model_selection import train_test_split

from train_utils import (
    best_group_split,
    summarize_partition,
    verify_arrays_finite,
    verify_no_duplicate_videos,
    verify_no_group_leakage,
    verify_split_shapes,
)
from training_config import (
    build_sample_metadata,
    get_config,
    get_event_split_path,
    parse_config,
    uses_shared_event_split,
)


TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Orden real del pipeline (fase 1):
#   01_preprocessing -> 02_temporal -> 03_split (solo split, SIN scaler)
#   -> 04_augmentation+scale (augment solo train + fit scaler en train)
#   -> 05_train. Este modulo SOLO genera el split con datos sin escalar.


def create_indices(labels, groups, config):
    indices = np.arange(len(labels))

    if config["split_strategy"] == "grouped":
        train_idx, temp_idx = best_group_split(
            labels,
            groups,
            VAL_SIZE + TEST_SIZE,
            config["random_seed"],
        )
        val_relative, test_relative = best_group_split(
            labels[temp_idx],
            groups[temp_idx],
            TEST_SIZE / (VAL_SIZE + TEST_SIZE),
            config["random_seed"] + 1000,
        )
        val_idx = temp_idx[val_relative]
        test_idx = temp_idx[test_relative]
    else:
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=VAL_SIZE + TEST_SIZE,
            stratify=labels,
            random_state=config["random_seed"],
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
            stratify=labels[temp_idx],
            random_state=config["random_seed"],
        )

    return train_idx, val_idx, test_idx


def print_partition(name, labels, groups):
    summarize_partition(name, labels, groups)


def load_normalized_arrays(config):
    data = np.load(config["normalized_path"])
    keypoints = np.asarray(data["keypoints"], dtype=np.float32)
    labels = np.asarray(data["labels"])
    video_ids = np.asarray(data["video_ids"]).astype(str)

    if "group_ids" in data.files and "source_views" in data.files:
        group_ids = np.asarray(data["group_ids"]).astype(str)
        source_views = np.asarray(data["source_views"]).astype(str)
    else:
        group_ids, source_views = build_sample_metadata(video_ids, config)

    return keypoints, labels, video_ids, group_ids, source_views


def build_common_event_partition(seed):
    """Particion comun de eventos fisicos usando left+right combinados.

    Junta todas las muestras disponibles de ambas vistas (con su label
    propio por vista) agrupadas por evento fisico y aplica best_group_split
    dos veces (train vs temp, val vs test). Asi left y right comparten
    exactamente los mismos eventos por particion cuando existe la pareja.
    """
    left_config = get_config("dl_left")
    right_config = get_config("dl_right")

    _, labels_left, _, groups_left, _ = load_normalized_arrays(left_config)
    _, labels_right, _, groups_right, _ = load_normalized_arrays(right_config)

    all_labels = np.concatenate([labels_left, labels_right])
    all_groups = np.concatenate([groups_left, groups_right])

    train_idx, temp_idx = best_group_split(
        all_labels,
        all_groups,
        VAL_SIZE + TEST_SIZE,
        seed,
    )
    val_relative, test_relative = best_group_split(
        all_labels[temp_idx],
        all_groups[temp_idx],
        TEST_SIZE / (VAL_SIZE + TEST_SIZE),
        seed + 1000,
    )
    val_idx = temp_idx[val_relative]
    test_idx = temp_idx[test_relative]

    unique_groups = np.unique(all_groups)
    partition_of = {}
    for index in train_idx:
        partition_of[str(all_groups[index])] = "train"
    for index in val_idx:
        partition_of[str(all_groups[index])] = "val"
    for index in test_idx:
        partition_of[str(all_groups[index])] = "test"

    assert len(partition_of) == len(unique_groups), (
        "Eventos sin particion asignada en el split comun"
    )
    verify_no_group_leakage(
        [event for event, part in partition_of.items() if part == "train"],
        [event for event, part in partition_of.items() if part == "val"],
        [event for event, part in partition_of.items() if part == "test"],
    )

    return partition_of


def paired_events_per_partition(partition_of):
    """Eventos disponibles SIMULTANEAMENTE en left y right por particion.

    No elimina muestras de los datasets originales; es solo para
    comparaciones pareadas entre vistas.
    """
    left_config = get_config("dl_left")
    right_config = get_config("dl_right")
    _, _, _, groups_left, _ = load_normalized_arrays(left_config)
    _, _, _, groups_right, _ = load_normalized_arrays(right_config)

    paired = {}

    for part in ("train", "val", "test"):
        events_left = {
            str(event) for event in groups_left
            if partition_of[str(event)] == part
        }
        events_right = {
            str(event) for event in groups_right
            if partition_of[str(event)] == part
        }
        paired[part] = np.asarray(sorted(events_left & events_right))

    assert set(paired["test"]) == (
        set(np.asarray(groups_left).astype(str))
        & set(np.asarray(groups_right).astype(str))
        & {e for e in partition_of if partition_of[e] == "test"}
    ), "paired_test inconsistente entre vistas"

    return paired


def save_common_event_split(partition_of, paired, seed):
    path = get_event_split_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    events = np.asarray(sorted(partition_of.keys()))
    partitions = np.asarray([partition_of[event] for event in events])
    np.savez_compressed(
        path,
        events=events,
        partitions=partitions,
        paired_train_events=paired["train"],
        paired_val_events=paired["val"],
        paired_test_events=paired["test"],
        seed=np.asarray([seed]),
    )
    print(f"Split comun de eventos guardado en: {path}")
    print(
        f"Paired (ambas vistas): train={len(paired['train'])}, "
        f"val={len(paired['val'])}, test={len(paired['test'])}"
    )
    return path


def indices_from_event_partition(group_ids, partition_of):
    group_ids = np.asarray(group_ids).astype(str)
    train_idx = np.where(
        np.asarray([partition_of[event] for event in group_ids]) == "train"
    )[0]
    val_idx = np.where(
        np.asarray([partition_of[event] for event in group_ids]) == "val"
    )[0]
    test_idx = np.where(
        np.asarray([partition_of[event] for event in group_ids]) == "test"
    )[0]

    assert len(train_idx) > 0 and len(val_idx) > 0 and len(test_idx) > 0, (
        "Particion vacia al mapear eventos al perfil actual"
    )
    return train_idx, val_idx, test_idx


def verify_split(X_train, X_val, X_test, y_train, y_val, y_test,
                 ids_train, ids_val, ids_test,
                 groups_train, groups_val, groups_test, config):
    verify_no_group_leakage(groups_train, groups_val, groups_test)
    verify_no_duplicate_videos(ids_train, ids_val, ids_test)
    verify_arrays_finite(
        np.asarray(X_train, dtype=np.float32),
        np.asarray(X_val, dtype=np.float32),
        np.asarray(X_test, dtype=np.float32),
        names=["X_train_split", "X_val_split", "X_test_split"],
    )
    verify_split_shapes(
        X_train, X_val, X_test,
        config["target_frames"], config["expected_keypoints"],
    )

    for name, labels in (
        ("train", y_train), ("val", y_val), ("test", y_test),
    ):
        labels = np.asarray(labels)
        assert set(np.unique(labels)).issubset({0, 1}), (
            f"Etiquetas no binarias en {name}"
        )
        assert (labels == 0).any() and (labels == 1).any(), (
            f"Particion {name} sin ambas clases"
        )

    print("Checks split: OK "
          "(sin leakage de eventos, sin videos duplicados, "
          "sin NaN/Inf, shapes correctas, ambas clases por particion)")


def split_and_scale(config):
    """Genera el split SIN escalar (el scaler vive en 04_augmentation)."""
    config["split_dir"].mkdir(parents=True, exist_ok=True)
    keypoints, labels, video_ids, group_ids, source_views = (
        load_normalized_arrays(config)
    )

    if uses_shared_event_split(config):
        partition_of = build_common_event_partition(config["random_seed"])
        paired = paired_events_per_partition(partition_of)
        save_common_event_split(
            partition_of, paired, config["random_seed"]
        )
        train_idx, val_idx, test_idx = indices_from_event_partition(
            group_ids, partition_of
        )
        print("Split comun por evento fisico (left/right compartido)")
    else:
        train_idx, val_idx, test_idx = create_indices(
            labels, group_ids, config
        )

    X_train = keypoints[train_idx]
    X_val = keypoints[val_idx]
    X_test = keypoints[test_idx]
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]

    samples, frames, points, channels = keypoints.shape
    print(f"Perfil: {config['profile']} ({config['display_name']})")
    print(
        f"Total: {samples} secuencias, {frames} frames, "
        f"{points} keypoints, {channels} features"
    )
    print(f"Distribucion: validos={sum(labels == 1)}, invalidos={sum(labels == 0)}")
    print()
    print_partition("Train", y_train, group_ids[train_idx])
    print_partition("Val", y_val, group_ids[val_idx])
    print_partition("Test", y_test, group_ids[test_idx])

    verify_split(
        X_train, X_val, X_test, y_train, y_val, y_test,
        video_ids[train_idx], video_ids[val_idx], video_ids[test_idx],
        group_ids[train_idx], group_ids[val_idx], group_ids[test_idx],
        config,
    )

    np.savez_compressed(
        config["split_path"],
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        ids_train=video_ids[train_idx],
        ids_val=video_ids[val_idx],
        ids_test=video_ids[test_idx],
        groups_train=group_ids[train_idx],
        groups_val=group_ids[val_idx],
        groups_test=group_ids[test_idx],
        views_train=source_views[train_idx],
        views_val=source_views[val_idx],
        views_test=source_views[test_idx],
        events_train=np.unique(group_ids[train_idx]),
        events_val=np.unique(group_ids[val_idx]),
        events_test=np.unique(group_ids[test_idx]),
        scaler_applied=np.asarray([False]),
    )

    print(f"\nSplit (sin escalar) guardado en: {config['split_path']}")
    print("Escalado pendiente: se aplica en 04_augmentation (fit solo en train).")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape:   {X_val.shape}")
    print(f"X_test shape:  {X_test.shape}")


if __name__ == "__main__":
    split_and_scale(parse_config("Split y escalado del dataset"))
