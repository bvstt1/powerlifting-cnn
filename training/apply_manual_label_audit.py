"""Aplica la auditoria manual de 41 eventos DL de forma reproducible.

Primero ejecutar --preview. Solo --apply crea backups y modifica los CSV.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from training_config import get_config, physical_event


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
LABEL_DIR = PROJECT_DIR / "etiquetado" / "dl"
DATA_DIR = SCRIPT_DIR / "data" / "dl"

LEFT_PATH = LABEL_DIR / "dl_left.csv"
RIGHT_PATH = LABEL_DIR / "dl_right.csv"
SIDE_PATH = LABEL_DIR / "dl_side.csv"
LEFT_BACKUP = LABEL_DIR / "dl_left_pre_manual_audit.csv"
RIGHT_BACKUP = LABEL_DIR / "dl_right_pre_manual_audit.csv"
SIDE_BACKUP = LABEL_DIR / "dl_side_pre_manual_audit.csv"


# physical_event, left_id, right_id, new_left, new_right, quality_flag
# Evento 235 corrige el typo 236 confirmado por el usuario.
CASES = [
    (80, 80, 80, 1, 1, ""),
    (87, 87, 87, 1, 1, ""),
    (165, 165, 165, 1, 1, ""),
    (170, 170, 170, 0, 0, ""),
    (171, 171, 171, 0, 1, ""),
    (172, 172, 172, 0, 1, ""),
    (173, 173, 173, 0, 0, ""),
    (174, 174, 174, 0, 1, ""),
    (175, 175, 175, 1, 1, ""),
    (177, 177, 177, 1, 1, ""),
    (184, 184, 184, 1, 1, ""),
    (196, 196, 196, 1, 1, "*L"),
    (197, 197, 197, 1, 1, "*L"),
    (205, 205, 205, 1, 1, ""),
    (235, 235, 235, 0, 0, ""),
    (240, 240, 240, 1, 1, ""),
    (243, 243, 243, 1, 1, "+'L"),
    (244, 244, 244, 0, 0, "+L"),
    (245, 245, 245, 0, 0, "+L"),
    (246, 246, 246, 0, 0, ""),
    (247, 247, 247, 0, 0, ""),
    (259, 259, 259, 0, 0, ""),
    (271, 271, 271, 0, 0, ""),
    (277, 277, 277, 1, 0, ""),
    (278, 278, 278, 1, 0, ""),
    (282, 282, 282, 1, 0, "*R"),
    (283, 283, 283, 1, 0, "*R"),
    (285, 285, 285, 1, 0, "*R"),
    (301, 301, 301, 0, 0, "*R"),
    (302, 302, 302, None, None, "DISCARD"),
    (312, 311, 312, 0, 0, ""),
    (316, 315, 316, 0, 0, ""),
    (329, 328, 329, 1, 1, ""),
    (335, 334, 335, 0, 0, ""),
    (336, 335, 336, 1, 1, "+'R"),
    (343, 342, 343, 1, 1, ""),
    (348, 347, 348, 0, 0, ""),
    (351, 350, 351, 1, 0, "+R"),
    (359, 358, 359, 1, 1, ""),
    (372, 371, 372, 1, 1, "*LR"),
    (373, 372, 373, 0, 0, "*R"),
]


def video_id(number):
    return f"dl_{number:03d}"


def one_row(df, value, name):
    rows = df.index[df["video_id"] == value].tolist()
    assert len(rows) == 1, f"{name}: esperado 1 registro para {value}, hay {len(rows)}"
    return rows[0]


def validate_identity(df, index, expected_id, event, view):
    row = df.loc[index]
    assert row["video_id"] == expected_id
    group_id, source_view = physical_event(expected_id, view)
    assert group_id == f"dl_event_{event:03d}"
    assert source_view == view
    archivo = row.get("archivo")

    if pd.notna(archivo):
        assert expected_id in str(archivo), (
            f"{view} evento {event}: archivo inesperado {archivo}"
        )


def flags_for(quality_flag):
    values = {
        "frame_loss_left": 0,
        "frame_loss_right": 0,
        "skeleton_issue_left": 0,
        "skeleton_issue_right": 0,
        "original_label_wrong_left": 0,
        "original_label_wrong_right": 0,
        "quality_flag_left": "",
        "quality_flag_right": "",
    }

    if quality_flag == "*L":
        values.update(frame_loss_left=1, quality_flag_left="*L")
    elif quality_flag == "*R":
        values.update(frame_loss_right=1, quality_flag_right="*R")
    elif quality_flag == "*LR":
        values.update(
            frame_loss_left=1,
            frame_loss_right=1,
            quality_flag_left="*L",
            quality_flag_right="*R",
        )
    elif quality_flag == "+L":
        values.update(skeleton_issue_left=1, quality_flag_left="+L")
    elif quality_flag == "+R":
        values.update(skeleton_issue_right=1, quality_flag_right="+R")
    elif quality_flag == "+'L":
        values.update(
            skeleton_issue_left=1,
            original_label_wrong_left=1,
            quality_flag_left="+'L",
        )
    elif quality_flag == "+'R":
        values.update(
            skeleton_issue_right=1,
            original_label_wrong_right=1,
            quality_flag_right="+'R",
        )

    return values


def max_nan_gap(array):
    maximum = 0

    for keypoint in range(array.shape[1]):
        for channel in range(array.shape[2]):
            mask = np.isnan(array[:, keypoint, channel])
            current = 0

            for missing in mask:
                current = current + 1 if missing else 0
                maximum = max(maximum, current)

    return maximum


def quality_metrics(event, view, source_id, quality_flag, discarded):
    from importlib import import_module

    preprocessing = import_module("01_preprocessing")
    config = get_config(f"dl_{view}")
    path = config["keypoints_dir"] / f"{source_id}.npy"
    keypoints = np.load(path).astype(np.float32)
    total_frames = int(keypoints.shape[0])
    initial_missing = np.isnan(keypoints[:, :, :2]).any(axis=(1, 2))
    cleaned = preprocessing.interpolate_small_gaps(keypoints)
    remaining_missing = np.isnan(cleaned[:, :, :2]).any(axis=(1, 2))
    interpolated_frames = int(sum(initial_missing & ~remaining_missing))
    valid_frames = int(total_frames - remaining_missing.sum())
    nan_ratio = float(remaining_missing.mean())
    confidence = keypoints[:, :, 2]
    finite_confidence = confidence[np.isfinite(confidence)]
    objective_reasons = []

    if keypoints.ndim != 3 or keypoints.shape[1:] != (2, 3):
        objective_reasons.append("invalid_shape")
    if total_frames < 10:
        objective_reasons.append("insufficient_frames")
    if np.isnan(keypoints).all():
        objective_reasons.append("all_nan")
    if np.isinf(keypoints).any():
        objective_reasons.append("inf_detected")
    if nan_ratio > 0.30:
        objective_reasons.append("nan_ratio_above_0.30")

    if discarded:
        status = "QUALITY_EXCLUDED"
        reason = "manual_review_discard"
    elif objective_reasons:
        status = "QUALITY_EXCLUDED"
        reason = "|".join(objective_reasons)
    elif quality_flag:
        status = "QUALITY_WARNING"
        reason = "manual_quality_flag_passes_objective_rules"
    else:
        status = "QUALITY_OK"
        reason = ""

    return {
        "physical_event": f"dl_event_{event:03d}",
        "view": view,
        "video_id": source_id,
        "quality_flag": quality_flag,
        "total_frames_original": total_frames,
        "valid_frames": valid_frames,
        "nan_ratio": nan_ratio,
        "n_interpolated_frames": interpolated_frames,
        "max_nan_gap": max_nan_gap(keypoints),
        "mean_keypoint_confidence": (
            float(finite_confidence.mean()) if len(finite_confidence) else np.nan
        ),
        "min_keypoint_confidence": (
            float(finite_confidence.min()) if len(finite_confidence) else np.nan
        ),
        "quality_status": status,
        "quality_reason": reason,
    }


def build_audit(left, right):
    audit_rows = []
    preview_rows = []
    quality_rows = []

    for event, left_number, right_number, new_left, new_right, quality_flag in CASES:
        left_id = video_id(left_number)
        right_id = video_id(right_number)
        left_index = one_row(left, left_id, "left")
        right_index = one_row(right, right_id, "right")
        validate_identity(left, left_index, left_id, event, "left")
        validate_identity(right, right_index, right_id, event, "right")
        old_left = int(left.loc[left_index, "label"])
        old_right = int(right.loc[right_index, "label"])
        discarded = event == 302
        flags = flags_for(quality_flag)
        preview_rows.append({
            "physical_event": f"dl_event_{event:03d}",
            "left_video_id": left_id,
            "right_video_id": right_id,
            "old_left_label": old_left,
            "new_left_label": "DISCARD" if discarded else new_left,
            "old_right_label": old_right,
            "new_right_label": "DISCARD" if discarded else new_right,
            "quality_flag": quality_flag,
        })
        audit_rows.append({
            "physical_event": f"dl_event_{event:03d}",
            "left_video_id": left_id,
            "right_video_id": right_id,
            "old_left_label": old_left,
            "new_left_label": "" if discarded else new_left,
            "old_right_label": old_right,
            "new_right_label": "" if discarded else new_right,
            "left_changed": int(not discarded and old_left != new_left),
            "right_changed": int(not discarded and old_right != new_right),
            "discarded": int(discarded),
            **flags,
            "review_notes": (
                "manual_review_discard" if discarded else "manual_label_audit"
            ),
        })
        quality_rows.append(quality_metrics(
            event, "left", left_id, flags["quality_flag_left"], discarded
        ))
        quality_rows.append(quality_metrics(
            event, "right", right_id, flags["quality_flag_right"], discarded
        ))

    assert len(audit_rows) == 41
    return pd.DataFrame(preview_rows), pd.DataFrame(audit_rows), pd.DataFrame(quality_rows)


def create_backup(source, backup):
    if backup.exists():
        raise FileExistsError(f"Backup ya existe; no se sobrescribe: {backup}")
    shutil.copy2(source, backup)


def set_label(df, source_id, label):
    index = one_row(df, source_id, "update")
    df.loc[index, "label"] = int(label)
    df.loc[index, "resultado"] = "valido" if int(label) == 1 else "invalido"


def apply_changes(left, right, side, audit):
    for df in (left, right, side):
        if "manual_exclude" not in df.columns:
            df["manual_exclude"] = 0
        if "exclude_reason" not in df.columns:
            df["exclude_reason"] = ""
        df["manual_exclude"] = df["manual_exclude"].fillna(0).astype(int)

    for event, left_number, right_number, new_left, new_right, _ in CASES:
        left_id = video_id(left_number)
        right_id = video_id(right_number)
        side_right_id = right_id
        side_left_id = video_id(413 + left_number)

        if event == 302:
            for df, source_id in (
                (left, left_id),
                (right, right_id),
                (side, side_right_id),
                (side, side_left_id),
            ):
                index = one_row(df, source_id, "manual discard")
                df.loc[index, "manual_exclude"] = 1
                df.loc[index, "exclude_reason"] = "manual_review_discard"
            continue

        set_label(left, left_id, new_left)
        set_label(right, right_id, new_right)
        set_label(side, side_left_id, new_left)
        set_label(side, side_right_id, new_right)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    left.to_csv(LEFT_PATH, index=False)
    right.to_csv(RIGHT_PATH, index=False)
    side.to_csv(SIDE_PATH, index=False)
    audit.to_csv(DATA_DIR / "manual_label_audit.csv", index=False)


def main(apply):
    left = pd.read_csv(LEFT_PATH)
    right = pd.read_csv(RIGHT_PATH)
    side = pd.read_csv(SIDE_PATH)
    preview, audit, quality = build_audit(left, right)
    print(preview.to_string(index=False))
    print(
        f"\nCambios previstos: left={int(audit['left_changed'].sum())}, "
        f"right={int(audit['right_changed'].sum())}, "
        f"descartados={int(audit['discarded'].sum())}"
    )

    if not apply:
        print("PREVIEW COMPLETADO: no se modifico ningun archivo.")
        return

    create_backup(LEFT_PATH, LEFT_BACKUP)
    create_backup(RIGHT_PATH, RIGHT_BACKUP)
    create_backup(SIDE_PATH, SIDE_BACKUP)
    apply_changes(left, right, side, audit)
    quality.to_csv(DATA_DIR / "manual_quality_audit.csv", index=False)
    print(f"Audit log: {DATA_DIR / 'manual_label_audit.csv'}")
    print(f"Quality log: {DATA_DIR / 'manual_quality_audit.csv'}")
    print("AUDITORIA APLICADA")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    assert args.preview != args.apply, "Elegir exactamente --preview o --apply"
    main(args.apply)
