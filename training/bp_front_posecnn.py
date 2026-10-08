"""Pipeline reproducible BP_FRONT PoseCNN monovista.

Etapas:
    python training/bp_front_posecnn.py --stage audit
    python training/bp_front_posecnn.py --stage prepare
    python training/bp_front_posecnn.py --stage train
    python training/bp_front_posecnn.py --stage cv
    python training/bp_front_posecnn.py --stage report
    python training/bp_front_posecnn.py --stage all

Este pipeline usa exclusivamente BP front. No usa group_id ni logica multivista.
La unica lectura de segmentacion es el manifest permitido para definir la
cohorte common; no se cargan mascaras, features ni modelos de segmentacion.
"""

import argparse
import copy
import csv
import hashlib
import json
import pickle
import platform
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

KEYPOINTS_DIR = PROJECT_ROOT / "keypoints" / "bp" / "front"
VIDEOS_DIR = PROJECT_ROOT / "dataset" / "bp" / "front"
LABELS_PATH = PROJECT_ROOT / "etiquetado" / "bp" / "bp_front.csv"
EXTRACTOR_PATH = (
    PROJECT_ROOT / "extract_keypoints" / "bp" / "extract_keypoints_front_bp.py"
)
POSE_MODEL_PATH = PROJECT_ROOT / "models" / "bp" / "bp_front_skeleton_v8.pt"
SEGMENTATION_MANIFEST_PATH = (
    PROJECT_ROOT
    / "segmentation" / "bp" / "front" / "manifests"
    / "bp_front_segmentation_manifest.csv"
)

DATA_DIR = SCRIPT_DIR / "data" / "bp" / "front"
MODEL_DIR = SCRIPT_DIR / "models" / "bp" / "front" / "posecnn"
RESULTS_DIR = SCRIPT_DIR / "results" / "bp" / "front" / "posecnn"
CV_MODEL_DIR = MODEL_DIR / "cv"
REPORT_PATH = PROJECT_ROOT / "Reporte_BP_FRONT_PoseCNN.md"

AUDIT_PATH = DATA_DIR / "bp_front_pose_audit.csv"
AUDIT_SUMMARY_PATH = DATA_DIR / "bp_front_pose_audit_summary.json"
RAW_SEQUENCES_PATH = DATA_DIR / "raw_sequences.npz"
PROCESSED_SEQUENCES_PATH = DATA_DIR / "processed_sequences.npz"
PREPROCESSING_MANIFEST_PATH = (
    DATA_DIR / "bp_front_pose_preprocessing_manifest.csv"
)
SPLIT_PATH = DATA_DIR / "bp_front_pose_split.csv"
SPLIT_ARRAYS_PATH = DATA_DIR / "bp_front_pose_split_arrays.npz"
SCALER_PATH = MODEL_DIR / "bp_front_pose_scaler.pkl"
CONFIG_PATH = MODEL_DIR / "bp_front_pose_config.json"
CHECKPOINT_PATH = MODEL_DIR / "bp_front_pose_best_model.pth"
HISTORY_PATH = RESULTS_DIR / "bp_front_pose_training_history.csv"
TEST_PREDICTIONS_PATH = RESULTS_DIR / "bp_front_pose_test_predictions.csv"
METRICS_PATH = RESULTS_DIR / "bp_front_pose_metrics.json"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "bp_front_pose_confusion_matrix.npy"
CV_RESULTS_PATH = RESULTS_DIR / "bp_front_pose_cv_results.csv"
CV_PREDICTIONS_PATH = RESULTS_DIR / "bp_front_pose_cv_predictions.csv"

SEED = 42
EXPERIMENT_REVISION = "BP_FRONT_POSECNN_BASELINE_V1_DEDUPLICATED"
TARGET_FRAMES = 170
EXPECTED_KEYPOINTS = 6
EXPECTED_CHANNELS = 3
MIN_FRAMES = 10
MAX_INTERPOLATION_GAP = 3
MAX_MISSING_XY_RATIO = 0.01
WARNING_MEDIAN_CONFIDENCE = 0.90
EXCLUDED_MEDIAN_CONFIDENCE = 0.50

BATCH_SIZE = 16
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 15
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
CLASSIFICATION_THRESHOLD = 0.5
AUGMENTATION_FACTOR = 3

# Orden operativo establecido por el skeleton del visualizador YOLO. El
# comentario historico del indice 5 repite "muneca derecha", pero la arista
# (0, 5) y la cadena anatomica bilateral identifican ese punto como izquierda.
KEYPOINT_SCHEMA = (
    (0, "left_elbow", "Codo izquierdo del levantador"),
    (1, "left_shoulder", "Hombro izquierdo del levantador"),
    (2, "right_shoulder", "Hombro derecho del levantador"),
    (3, "right_elbow", "Codo derecho del levantador"),
    (4, "right_wrist", "Muneca derecha del levantador"),
    (5, "left_wrist", "Muneca izquierda del levantador"),
)
LEFT_SHOULDER_INDEX = 1
RIGHT_SHOULDER_INDEX = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_output_dirs():
    for path in (DATA_DIR, MODEL_DIR, RESULTS_DIR, CV_MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, data):
    path.write_text(
        json.dumps(json_ready(data), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def longest_true_run(mask):
    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def sorted_ids(paths):
    return [path.stem for path in sorted(paths)]


def audit_quality(shape_ok, frames, inf_count, nan_ratio_xy,
                  longest_missing_gap, completely_missing_frames,
                  median_confidence, confidence_out_of_range,
                  shoulder_scale):
    reasons = []
    warnings = []

    if not shape_ok:
        reasons.append("invalid_shape")
    if frames is not None and frames < MIN_FRAMES:
        reasons.append("too_few_frames")
    if inf_count > 0:
        reasons.append("contains_inf")
    if confidence_out_of_range > 0:
        reasons.append("confidence_out_of_range")
    if nan_ratio_xy > MAX_MISSING_XY_RATIO:
        reasons.append("missing_xy_ratio_above_1pct")
    if longest_missing_gap > MAX_INTERPOLATION_GAP:
        reasons.append("missing_gap_above_3_frames")
    if completely_missing_frames > MAX_INTERPOLATION_GAP:
        reasons.append("long_completely_missing_region")
    if median_confidence is not None and median_confidence < EXCLUDED_MEDIAN_CONFIDENCE:
        reasons.append("median_confidence_below_0.50")
    if shoulder_scale is not None and shoulder_scale <= 1e-6:
        reasons.append("invalid_intershoulder_scale")

    if reasons:
        return "POSE_QUALITY_EXCLUDED", "|".join(reasons)

    if nan_ratio_xy > 0:
        warnings.append("small_interpolable_missing_gap")
    if completely_missing_frames > 0:
        warnings.append("completely_missing_frames_interpolable")
    if (
        median_confidence is not None
        and median_confidence < WARNING_MEDIAN_CONFIDENCE
    ):
        warnings.append("median_confidence_below_0.90")

    if warnings:
        return "POSE_QUALITY_WARNING", "|".join(warnings)
    return "POSE_QUALITY_OK", ""


def temporal_distribution(lengths):
    values = np.asarray(lengths, dtype=np.float64)
    percentiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    return {
        "N": int(len(values)),
        "min": int(values.min()),
        "P1": float(percentiles[0]),
        "P5": float(percentiles[1]),
        "P25": float(percentiles[2]),
        "median": float(percentiles[3]),
        "P75": float(percentiles[4]),
        "P95": float(percentiles[5]),
        "P99": float(percentiles[6]),
        "max": int(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
    }


def inspect_pose_file(path):
    row = {
        "video_id": path.stem,
        "pose_path": str(path),
        "extension": path.suffix,
        "load_ok": False,
        "ndim": None,
        "shape": "",
        "T_original": None,
        "K": None,
        "C": None,
        "dtype": "",
        "pose_content_sha256": "",
        "nan_count": None,
        "nan_ratio_total": None,
        "nan_ratio_xy": None,
        "inf_count": None,
        "valid_frames": None,
        "invalid_frames": None,
        "n_frames_with_missing_keypoints": None,
        "n_completely_missing_frames": None,
        "mean_confidence": None,
        "median_confidence": None,
        "min_confidence": None,
        "max_confidence": None,
        "longest_missing_gap": None,
        "confidence_out_of_range": None,
        "coordinates_out_of_unit_range": None,
        "median_intershoulder_distance": None,
        "structural_anomaly": "",
    }

    try:
        sequence = np.load(path, allow_pickle=False)
    except Exception as error:
        row["structural_anomaly"] = f"load_error:{type(error).__name__}"
        row["quality_status"] = "POSE_QUALITY_EXCLUDED"
        row["quality_reasons"] = row["structural_anomaly"]
        return row

    row["load_ok"] = True
    row["ndim"] = int(sequence.ndim)
    row["shape"] = str(tuple(sequence.shape))
    row["dtype"] = str(sequence.dtype)
    content_digest = hashlib.sha256()
    content_digest.update(str(sequence.shape).encode("ascii"))
    content_digest.update(str(sequence.dtype).encode("ascii"))
    content_digest.update(np.ascontiguousarray(sequence).tobytes())
    row["pose_content_sha256"] = content_digest.hexdigest()

    if sequence.ndim != 3:
        row["structural_anomaly"] = "expected_3_dimensions"
        row["quality_status"] = "POSE_QUALITY_EXCLUDED"
        row["quality_reasons"] = row["structural_anomaly"]
        return row

    frames, keypoints, channels = sequence.shape
    row["T_original"] = int(frames)
    row["K"] = int(keypoints)
    row["C"] = int(channels)
    shape_ok = keypoints == EXPECTED_KEYPOINTS and channels == EXPECTED_CHANNELS

    row["nan_count"] = int(np.isnan(sequence).sum())
    row["nan_ratio_total"] = float(np.isnan(sequence).mean())
    row["inf_count"] = int(np.isinf(sequence).sum())

    if channels >= 2:
        xy = sequence[..., :2]
        missing_keypoint = np.isnan(xy).any(axis=2)
        frame_missing = missing_keypoint.any(axis=1)
        complete_missing = missing_keypoint.all(axis=1)
        row["nan_ratio_xy"] = float(np.isnan(xy).mean())
        row["valid_frames"] = int((~frame_missing).sum())
        row["invalid_frames"] = int(frame_missing.sum())
        row["n_frames_with_missing_keypoints"] = int(frame_missing.sum())
        row["n_completely_missing_frames"] = int(complete_missing.sum())
        row["longest_missing_gap"] = longest_true_run(frame_missing)
        finite_xy = xy[np.isfinite(xy)]
        row["coordinates_out_of_unit_range"] = int(
            ((finite_xy < 0) | (finite_xy > 1)).sum()
        )
    else:
        row["nan_ratio_xy"] = 1.0
        row["valid_frames"] = 0
        row["invalid_frames"] = int(frames)
        row["n_frames_with_missing_keypoints"] = int(frames)
        row["n_completely_missing_frames"] = int(frames)
        row["longest_missing_gap"] = int(frames)
        row["coordinates_out_of_unit_range"] = 0

    median_confidence = None
    confidence_out_of_range = 0
    if channels >= 3:
        confidence = sequence[..., 2]
        finite_confidence = confidence[np.isfinite(confidence)]
        if finite_confidence.size:
            row["mean_confidence"] = float(finite_confidence.mean())
            row["median_confidence"] = float(np.median(finite_confidence))
            row["min_confidence"] = float(finite_confidence.min())
            row["max_confidence"] = float(finite_confidence.max())
            confidence_out_of_range = int(
                ((finite_confidence < 0) | (finite_confidence > 1)).sum()
            )
            median_confidence = row["median_confidence"]
    row["confidence_out_of_range"] = confidence_out_of_range

    shoulder_scale = None
    if shape_ok and frames:
        shoulders = sequence[:, [LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX], :2]
        valid = np.isfinite(shoulders).all(axis=(1, 2))
        if valid.any():
            distances = np.linalg.norm(
                shoulders[valid, 0] - shoulders[valid, 1], axis=1
            )
            shoulder_scale = float(np.median(distances))
            row["median_intershoulder_distance"] = shoulder_scale

    quality_status, quality_reasons = audit_quality(
        shape_ok=shape_ok,
        frames=frames,
        inf_count=row["inf_count"],
        nan_ratio_xy=row["nan_ratio_xy"],
        longest_missing_gap=row["longest_missing_gap"],
        completely_missing_frames=row["n_completely_missing_frames"],
        median_confidence=median_confidence,
        confidence_out_of_range=confidence_out_of_range,
        shoulder_scale=shoulder_scale,
    )
    if not shape_ok:
        row["structural_anomaly"] = "unexpected_K_or_C"
    row["quality_status"] = quality_status
    row["quality_reasons"] = quality_reasons
    return row


def decoded_video_sha256(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir para hash decodificado: {path}")
    digest = hashlib.sha256()
    frames = 0
    try:
        while True:
            read_ok, frame = capture.read()
            if not read_ok:
                break
            digest.update(np.ascontiguousarray(frame).tobytes())
            frames += 1
    finally:
        capture.release()
    return digest.hexdigest(), frames


def run_audit():
    ensure_output_dirs()

    required_paths = (
        KEYPOINTS_DIR,
        VIDEOS_DIR,
        LABELS_PATH,
        EXTRACTOR_PATH,
        POSE_MODEL_PATH,
        SEGMENTATION_MANIFEST_PATH,
    )
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise RuntimeError(f"Rutas requeridas inexistentes: {missing_paths}")

    pose_files = sorted(KEYPOINTS_DIR.glob("*.npy"))
    video_files = sorted(VIDEOS_DIR.glob("*.mp4"))
    labels = pd.read_csv(LABELS_PATH, dtype={"video_id": str})
    segmentation = pd.read_csv(SEGMENTATION_MANIFEST_PATH, dtype={"video_id": str})

    pose_ids = sorted_ids(pose_files)
    video_ids = sorted_ids(video_files)
    csv_ids = labels["video_id"].astype(str).tolist()
    segmentation_ids = segmentation["video_id"].astype(str).tolist()

    model = YOLO(str(POSE_MODEL_PATH))
    model_kpt_shape = model.model.yaml.get("kpt_shape", [])

    audit_rows = [inspect_pose_file(path) for path in pose_files]
    labels_by_id = labels.set_index("video_id", drop=False)
    segmentation_by_id = segmentation.set_index("video_id", drop=False)

    valid_segmentation_ids = set(
        segmentation.loc[
            segmentation["quality_status"].isin(["QUALITY_OK", "QUALITY_WARNING"])
            & (segmentation["status"] != "ERROR"),
            "video_id",
        ].astype(str)
    )

    for row in audit_rows:
        video_id = row["video_id"]
        row["video_exists"] = video_id in set(video_ids)
        row["csv_exists"] = video_id in set(csv_ids)
        row["segmentation_valid"] = video_id in valid_segmentation_ids
        if video_id in labels_by_id.index:
            label_row = labels_by_id.loc[video_id]
            row["label"] = int(label_row["label"])
            row["resultado"] = str(label_row["resultado"])
            row["codigos"] = "" if pd.isna(label_row["codigos"]) else str(label_row["codigos"])
        else:
            row["label"] = None
            row["resultado"] = ""
            row["codigos"] = ""

    audit = pd.DataFrame(audit_rows)
    audit["duplicate_pose_content_of"] = ""
    duplicate_content_groups = []
    duplicated_hashes = audit.loc[
        audit["pose_content_sha256"].duplicated(keep=False),
        "pose_content_sha256",
    ].unique()
    for content_hash in duplicated_hashes:
        duplicate_ids = sorted(
            audit.loc[
                audit["pose_content_sha256"] == content_hash, "video_id"
            ].astype(str).tolist()
        )
        canonical_id = duplicate_ids[0]
        decoded_hashes = {}
        decoded_frames = {}
        for video_id in duplicate_ids:
            decoded_hash, frame_count = decoded_video_sha256(
                VIDEOS_DIR / f"{video_id}.mp4"
            )
            decoded_hashes[video_id] = decoded_hash
            decoded_frames[video_id] = frame_count
        duplicate_content_groups.append({
            "canonical_id": canonical_id,
            "duplicate_ids": duplicate_ids[1:],
            "all_ids": duplicate_ids,
            "pose_content_sha256": content_hash,
            "decoded_video_identical": len(set(decoded_hashes.values())) == 1,
            "decoded_video_sha256": decoded_hashes,
            "decoded_frames": decoded_frames,
        })
        for duplicate_id in duplicate_ids[1:]:
            mask = audit["video_id"] == duplicate_id
            reason = f"duplicate_pose_content_of_{canonical_id}"
            audit.loc[mask, "duplicate_pose_content_of"] = canonical_id
            audit.loc[mask, "quality_status"] = "POSE_QUALITY_EXCLUDED"
            audit.loc[mask, "quality_reasons"] = reason
    audit.to_csv(AUDIT_PATH, index=False, encoding="utf-8")

    pose_set = set(pose_ids)
    video_set = set(video_ids)
    csv_set = set(csv_ids)
    triple = video_set & csv_set & pose_set

    labels_numeric = pd.to_numeric(labels["label"], errors="coerce")
    result_normalized = labels["resultado"].astype(str).str.strip().str.lower()
    result_label_mismatches = labels.loc[
        ((result_normalized == "valido") & (labels_numeric != 1))
        | ((result_normalized == "invalido") & (labels_numeric != 0)),
        "video_id",
    ].astype(str).tolist()

    usable_pose_ids = set(
        audit.loc[audit["quality_status"] != "POSE_QUALITY_EXCLUDED", "video_id"]
    )
    common_ids = usable_pose_ids & valid_segmentation_ids
    lengths = audit.loc[audit["T_original"].notna(), "T_original"].astype(int)

    summary = {
        "dataset": {
            "keypoints_path": str(KEYPOINTS_DIR),
            "extension": ".npy",
            "pose_files": len(pose_files),
            "pose_id_min": min(pose_ids) if pose_ids else None,
            "pose_id_max": max(pose_ids) if pose_ids else None,
            "pose_naming": "bp_NNN.npy",
            "videos": len(video_files),
            "video_id_min": min(video_ids) if video_ids else None,
            "video_id_max": max(video_ids) if video_ids else None,
            "csv_rows": int(len(labels)),
            "csv_unique_ids": int(labels["video_id"].nunique()),
            "segmentation_manifest_rows": int(len(segmentation)),
        },
        "pose_model": {
            "path": str(POSE_MODEL_PATH),
            "version": "bp_front_skeleton_v8.pt",
            "sha256": file_sha256(POSE_MODEL_PATH),
            "kpt_shape": model_kpt_shape,
            "extractor": str(EXTRACTOR_PATH),
            "channels": ["x_normalized", "y_normalized", "confidence"],
            "keypoint_schema": [
                {"index": index, "keypoint_name": name, "description": description}
                for index, name, description in KEYPOINT_SCHEMA
            ],
        },
        "pose_integrity": {
            "shape_counts": audit["shape"].value_counts(dropna=False).to_dict(),
            "dtype_counts": audit["dtype"].value_counts(dropna=False).to_dict(),
            "quality_counts": audit["quality_status"].value_counts().to_dict(),
            "total_nan": int(audit["nan_count"].fillna(0).sum()),
            "total_inf": int(audit["inf_count"].fillna(0).sum()),
            "total_frames_with_missing": int(
                audit["n_frames_with_missing_keypoints"].fillna(0).sum()
            ),
            "total_completely_missing_frames": int(
                audit["n_completely_missing_frames"].fillna(0).sum()
            ),
            "global_min_confidence": float(audit["min_confidence"].min()),
            "global_mean_confidence": float(
                np.average(
                    audit["mean_confidence"],
                    weights=audit["T_original"] * EXPECTED_KEYPOINTS,
                )
            ),
            "duplicate_content_groups": duplicate_content_groups,
        },
        "temporal_distribution": temporal_distribution(lengths),
        "labels": {
            "path": str(LABELS_PATH),
            "columns": labels.columns.tolist(),
            "rows": int(len(labels)),
            "unique_ids": int(labels["video_id"].nunique()),
            "duplicate_ids": sorted(
                labels.loc[labels["video_id"].duplicated(keep=False), "video_id"]
                .astype(str).unique().tolist()
            ),
            "missing_by_column": labels.isna().sum().astype(int).to_dict(),
            "label_counts": labels_numeric.value_counts(dropna=False).to_dict(),
            "valid_count": int((labels_numeric == 1).sum()),
            "invalid_count": int((labels_numeric == 0).sum()),
            "valid_percent": float((labels_numeric == 1).mean() * 100),
            "invalid_percent": float((labels_numeric == 0).mean() * 100),
            "missing_labels": int(labels_numeric.isna().sum()),
            "labels_outside_binary": sorted(
                labels_numeric.loc[~labels_numeric.isin([0, 1]) & labels_numeric.notna()]
                .unique().tolist()
            ),
            "resultado_label_mismatches": result_label_mismatches,
            "mapping": {"0": "invalido", "1": "valido"},
            "mapping_evidence": [
                "bp_front.csv: resultado valido siempre coincide con label 1",
                "bp_front.csv: resultado invalido siempre coincide con label 0",
                "training/01_preprocessing.py acepta exclusivamente labels 0/1",
                "AGENTS.md documenta 1=valido y 0=invalido",
            ],
        },
        "correspondence": {
            "video_csv_pose_intersection": len(triple),
            "video_without_csv": sorted(video_set - csv_set),
            "video_without_pose": sorted(video_set - pose_set),
            "csv_without_video": sorted(csv_set - video_set),
            "csv_without_pose": sorted(csv_set - pose_set),
            "pose_without_video": sorted(pose_set - video_set),
            "pose_without_csv": sorted(pose_set - csv_set),
            "duplicate_video_ids": [],
            "duplicate_pose_ids": [],
            "duplicate_pose_content_groups": duplicate_content_groups,
            "duplicate_csv_ids": sorted(
                labels.loc[labels["video_id"].duplicated(keep=False), "video_id"]
                .astype(str).unique().tolist()
            ),
        },
        "cohorts": {
            "pose_valid": len(usable_pose_ids),
            "segmentation_valid": len(valid_segmentation_ids),
            "pose_only": len(usable_pose_ids),
            "common_pose_segmentation": len(common_ids),
            "segmentation_quality_error_ids": sorted(
                segmentation.loc[
                    segmentation["quality_status"] == "QUALITY_ERROR", "video_id"
                ].astype(str).tolist()
            ),
        },
        "quality_rules": {
            "minimum_frames": MIN_FRAMES,
            "max_interpolation_gap_frames": MAX_INTERPOLATION_GAP,
            "max_missing_xy_ratio": MAX_MISSING_XY_RATIO,
            "inf_allowed": False,
            "expected_shape": ["T", EXPECTED_KEYPOINTS, EXPECTED_CHANNELS],
            "warning_median_confidence": WARNING_MEDIAN_CONFIDENCE,
            "excluded_median_confidence": EXCLUDED_MEDIAN_CONFIDENCE,
            "exclude_noncanonical_exact_duplicates": True,
            "justification": (
                "BP_FRONT actual no contiene missing ni Inf; gap<=3 equivale a "
                "100 ms a 30 FPS y se conserva solo como tolerancia conservadora. "
                "No se adopta el umbral historico de 30%."
            ),
        },
    }

    blockers = []
    if model_kpt_shape != [EXPECTED_KEYPOINTS, EXPECTED_CHANNELS]:
        blockers.append(f"model_kpt_shape={model_kpt_shape}")
    if labels["video_id"].duplicated().any():
        blockers.append("duplicate_csv_ids")
    if labels_numeric.isna().any() or not labels_numeric.isin([0, 1]).all():
        blockers.append("invalid_or_missing_labels")
    if result_label_mismatches:
        blockers.append("resultado_label_contradiction")
    if len(triple) != len(pose_files) or len(triple) != len(labels):
        blockers.append("video_csv_pose_mismatch")
    if not {0, 1}.issubset(set(labels_numeric.astype(int))):
        blockers.append("both_classes_not_present")
    if not usable_pose_ids:
        blockers.append("no_usable_pose")

    summary["critical_blockers"] = blockers
    save_json(AUDIT_SUMMARY_PATH, summary)

    print("\nAUDITORIA BP_FRONT POSE")
    print(f"Pose files: {len(pose_files)}")
    print(f"CSV rows: {len(labels)}")
    print(f"Videos: {len(video_files)}")
    print(f"Pose utilizables: {len(usable_pose_ids)}")
    print(f"Pose excluidos: {len(pose_files) - len(usable_pose_ids)}")
    print(f"Pose-only: {len(usable_pose_ids)}")
    print(f"Common Pose+Seg: {len(common_ids)}")
    print(f"Temporal: {summary['temporal_distribution']}")
    print(f"Quality: {summary['pose_integrity']['quality_counts']}")
    print(f"Blockers: {blockers or 'ninguno'}")
    return summary


def interpolate_small_gaps(sequence):
    result = sequence.copy().astype(np.float32)
    interpolated_values = 0
    frames = len(result)

    for keypoint in range(result.shape[1]):
        for channel in range(result.shape[2]):
            values = result[:, keypoint, channel]
            missing = ~np.isfinite(values)
            index = 0
            while index < frames:
                if not missing[index]:
                    index += 1
                    continue
                start = index
                while index < frames and missing[index]:
                    index += 1
                end = index
                gap = end - start
                if (
                    gap <= MAX_INTERPOLATION_GAP
                    and start > 0
                    and end < frames
                    and np.isfinite(values[start - 1])
                    and np.isfinite(values[end])
                ):
                    values[start:end] = np.linspace(
                        values[start - 1], values[end], gap + 2
                    )[1:-1]
                    interpolated_values += gap
            result[:, keypoint, channel] = values
    return result, interpolated_values


def normalize_spatial_bp_front(sequence):
    xy = sequence[..., :2]
    shoulders = xy[:, [LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX]]
    valid = np.isfinite(shoulders).all(axis=(1, 2))
    if not valid.any():
        raise ValueError("No hay frames validos para normalizacion por hombros")

    shoulder_midpoints = shoulders[valid].mean(axis=1)
    center = np.median(shoulder_midpoints, axis=0)
    distances = np.linalg.norm(
        shoulders[valid, 0] - shoulders[valid, 1], axis=1
    )
    scale = float(np.median(distances))
    if not np.isfinite(scale) or scale <= 1e-6:
        raise ValueError(f"Escala interhombros invalida: {scale}")

    normalized = sequence.copy().astype(np.float32)
    normalized[..., :2] = (xy - center) / scale
    return normalized, center.astype(np.float32), scale


def resample_temporal(sequence, target_frames=TARGET_FRAMES):
    source_frames = len(sequence)
    if source_frames < 2:
        raise ValueError("Se requieren al menos 2 frames para resampling")
    source_time = np.linspace(0.0, 1.0, source_frames)
    target_time = np.linspace(0.0, 1.0, target_frames)
    output = np.empty(
        (target_frames, sequence.shape[1], sequence.shape[2]),
        dtype=np.float32,
    )
    for keypoint in range(sequence.shape[1]):
        for channel in range(sequence.shape[2]):
            values = sequence[:, keypoint, channel]
            if not np.isfinite(values).all():
                raise ValueError("Resampling recibio valores no finitos")
            output[:, keypoint, channel] = np.interp(
                target_time, source_time, values
            )
    return output


def stratified_three_way_split(ids, labels):
    ids = np.asarray(ids).astype(str)
    labels = np.asarray(labels).astype(int)
    indices = np.arange(len(ids))

    first = StratifiedShuffleSplit(
        n_splits=1, test_size=0.15, random_state=SEED
    )
    development_idx, test_idx = next(first.split(indices, labels))

    validation_fraction_of_development = 0.15 / 0.85
    second = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_fraction_of_development,
        random_state=SEED,
    )
    train_local, val_local = next(
        second.split(development_idx, labels[development_idx])
    )
    train_idx = development_idx[train_local]
    val_idx = development_idx[val_local]

    assignment = {}
    for split_name, split_indices in (
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx),
    ):
        for index in split_indices:
            assignment[ids[index]] = split_name
    return assignment


def prepare_data():
    summary = run_audit()
    if summary["critical_blockers"]:
        raise RuntimeError(
            f"Auditoria bloqueante: {summary['critical_blockers']}"
        )

    audit = pd.read_csv(AUDIT_PATH, dtype={"video_id": str})
    labels = pd.read_csv(LABELS_PATH, dtype={"video_id": str})
    labels_by_id = labels.set_index("video_id")
    pose_files = sorted(KEYPOINTS_DIR.glob("*.npy"))

    raw_ids = []
    raw_sequences = []
    for path in pose_files:
        raw_ids.append(path.stem)
        raw_sequences.append(np.load(path, allow_pickle=False).astype(np.float32))
    raw_object = np.empty(len(raw_sequences), dtype=object)
    raw_object[:] = raw_sequences
    np.savez_compressed(
        RAW_SEQUENCES_PATH,
        video_ids=np.asarray(raw_ids),
        sequences=raw_object,
    )

    audit_by_id = audit.set_index("video_id")
    processed_ids = []
    processed_labels = []
    spatial_sequences = []
    temporal_sequences = []
    manifest_rows = []

    for path in pose_files:
        video_id = path.stem
        audit_row = audit_by_id.loc[video_id]
        cleaned = None
        manifest = {
            "video_id": video_id,
            "label": int(labels_by_id.loc[video_id, "label"]),
            "quality_status": audit_row["quality_status"],
            "quality_reasons": "" if pd.isna(audit_row["quality_reasons"]) else audit_row["quality_reasons"],
            "T_original": int(audit_row["T_original"]),
            "K": int(audit_row["K"]),
            "C": int(audit_row["C"]),
            "interpolation_max_gap": MAX_INTERPOLATION_GAP,
            "interpolated_values": 0,
            "remaining_nonfinite_values": None,
            "spatial_center_x": None,
            "spatial_center_y": None,
            "spatial_scale_intershoulder": None,
            "target_frames": TARGET_FRAMES,
            "included": False,
            "exclusion_reason": "",
        }

        if audit_row["quality_status"] == "POSE_QUALITY_EXCLUDED":
            manifest["exclusion_reason"] = audit_row["quality_reasons"]
            manifest_rows.append(manifest)
            continue

        try:
            sequence = np.load(path, allow_pickle=False).astype(np.float32)
            cleaned, interpolated_values = interpolate_small_gaps(sequence)
            remaining_nonfinite = int((~np.isfinite(cleaned)).sum())
            if remaining_nonfinite:
                raise ValueError("missing_values_remain_after_small_gap_interpolation")
            normalized, center, scale = normalize_spatial_bp_front(cleaned)
            temporal = resample_temporal(normalized)
            if not np.isfinite(temporal).all():
                raise ValueError("nonfinite_after_temporal_resampling")
        except Exception as error:
            manifest["exclusion_reason"] = str(error)
            manifest["remaining_nonfinite_values"] = int(
                (~np.isfinite(cleaned)).sum()
            ) if cleaned is not None else None
            manifest_rows.append(manifest)
            continue

        manifest.update({
            "interpolated_values": int(interpolated_values),
            "remaining_nonfinite_values": remaining_nonfinite,
            "spatial_center_x": float(center[0]),
            "spatial_center_y": float(center[1]),
            "spatial_scale_intershoulder": float(scale),
            "included": True,
        })
        manifest_rows.append(manifest)
        processed_ids.append(video_id)
        processed_labels.append(int(labels_by_id.loc[video_id, "label"]))
        spatial_sequences.append(normalized)
        temporal_sequences.append(temporal)

    preprocessing_manifest = pd.DataFrame(manifest_rows)
    preprocessing_manifest.to_csv(
        PREPROCESSING_MANIFEST_PATH, index=False, encoding="utf-8"
    )

    spatial_object = np.empty(len(spatial_sequences), dtype=object)
    spatial_object[:] = spatial_sequences
    temporal_array = np.stack(temporal_sequences).astype(np.float32)
    processed_ids = np.asarray(processed_ids)
    processed_labels = np.asarray(processed_labels, dtype=np.int64)
    np.savez_compressed(
        PROCESSED_SEQUENCES_PATH,
        video_ids=processed_ids,
        labels=processed_labels,
        spatial_sequences=spatial_object,
        sequences=temporal_array,
    )

    segmentation = pd.read_csv(
        SEGMENTATION_MANIFEST_PATH, dtype={"video_id": str}
    )
    valid_segmentation_ids = set(
        segmentation.loc[
            segmentation["quality_status"].isin(["QUALITY_OK", "QUALITY_WARNING"])
            & (segmentation["status"] != "ERROR"),
            "video_id",
        ].astype(str)
    )

    pose_assignment = stratified_three_way_split(
        processed_ids, processed_labels
    )
    common_mask = np.asarray([
        video_id in valid_segmentation_ids for video_id in processed_ids
    ])
    common_assignment = stratified_three_way_split(
        processed_ids[common_mask], processed_labels[common_mask]
    )

    split_rows = []
    for video_id, label in zip(processed_ids, processed_labels):
        split_rows.append({
            "video_id": video_id,
            "label": int(label),
            "pose_only_member": True,
            "pose_only_split": pose_assignment[video_id],
            "common_member": video_id in valid_segmentation_ids,
            "common_split": common_assignment.get(video_id, "not_in_common"),
            "split_strategy": "stratified_by_sample",
            "seed": SEED,
        })
    split = pd.DataFrame(split_rows).sort_values("video_id")
    split.to_csv(SPLIT_PATH, index=False, encoding="utf-8")

    split_indices = {}
    for split_name in ("train", "validation", "test"):
        split_ids = set(
            split.loc[split["pose_only_split"] == split_name, "video_id"]
        )
        split_indices[split_name] = np.asarray([
            index for index, video_id in enumerate(processed_ids)
            if video_id in split_ids
        ], dtype=np.int64)

    train_idx = split_indices["train"]
    val_idx = split_indices["validation"]
    test_idx = split_indices["test"]
    scaler = StandardScaler()
    scaler.fit(temporal_array[train_idx].reshape(len(train_idx), -1))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCALER_PATH, "wb") as file:
        pickle.dump(scaler, file)

    X_train_original = temporal_array[train_idx]
    X_val = scaler.transform(
        temporal_array[val_idx].reshape(len(val_idx), -1)
    ).reshape(temporal_array[val_idx].shape).astype(np.float32)
    X_test = scaler.transform(
        temporal_array[test_idx].reshape(len(test_idx), -1)
    ).reshape(temporal_array[test_idx].shape).astype(np.float32)
    X_train = scaler.transform(
        X_train_original.reshape(len(train_idx), -1)
    ).reshape(X_train_original.shape).astype(np.float32)

    np.savez_compressed(
        SPLIT_ARRAYS_PATH,
        X_train_original=X_train_original,
        X_train=X_train,
        y_train=processed_labels[train_idx],
        ids_train=processed_ids[train_idx],
        X_validation=X_val,
        y_validation=processed_labels[val_idx],
        ids_validation=processed_ids[val_idx],
        X_test=X_test,
        y_test=processed_labels[test_idx],
        ids_test=processed_ids[test_idx],
        scaler_fit_source=np.asarray(["train_original_preprocessed_only"]),
    )

    print_safety_checkpoint()
    return temporal_array, processed_labels, processed_ids, split


def class_counts(labels):
    labels = np.asarray(labels).astype(int)
    return {
        "invalid_0": int((labels == 0).sum()),
        "valid_1": int((labels == 1).sum()),
    }


def print_safety_checkpoint():
    summary = json.loads(AUDIT_SUMMARY_PATH.read_text(encoding="utf-8"))
    split = pd.read_csv(SPLIT_PATH)
    data = np.load(PROCESSED_SEQUENCES_PATH, allow_pickle=True)

    print("\n" + "=" * 68)
    print("CHECKPOINT DE SEGURIDAD BP_FRONT POSECNN")
    print("=" * 68)
    print("BP_FRONT MONOVISTA = confirmado")
    print(f"Videos originales: {summary['dataset']['videos']}")
    print(f"CSV rows: {summary['dataset']['csv_rows']}")
    print(f"Pose files: {summary['dataset']['pose_files']}")
    print(f"Pose utilizables: {summary['cohorts']['pose_valid']}")
    print(
        "Pose excluidos: "
        f"{summary['dataset']['pose_files'] - summary['cohorts']['pose_valid']}"
    )
    print(f"Cohorte Pose-only: {summary['cohorts']['pose_only']}")
    print(
        "Cohorte Common Pose+Seg: "
        f"{summary['cohorts']['common_pose_segmentation']}"
    )
    print("Label mapping: 0 = invalido | 1 = valido")

    for split_name in ("train", "validation", "test"):
        subset = split.loc[split["pose_only_split"] == split_name]
        counts = class_counts(subset["label"])
        print(f"{split_name.title()}: {len(subset)} | {counts}")

    print(
        f"Input shape: T={data['sequences'].shape[1]}, "
        f"K={data['sequences'].shape[2]}, C={data['sequences'].shape[3]}"
    )
    print(
        "Normalizacion espacial: centro mediano de secuencia del punto medio "
        "de hombros; escala mediana interhombros"
    )
    print("Normalizacion temporal: resampling lineal a T=170")
    print("Scaler fit source: TRAIN ORIGINAL PREPROCESADO")
    print(
        "Augmentation: train only, shift=0.05, scale=0.9-1.1, "
        "noise=0.01, rotation=10 grados, temporal shift=3, sin flip"
    )
    print("Split strategy: STRATIFIED BY SAMPLE")
    print("Cross-validation strategy: STRATIFIED K-FOLD")
    print("NO GROUP_ID")
    print("NO MULTIVIEW")
    print("=" * 68 + "\n")

    if summary["critical_blockers"]:
        raise RuntimeError(
            f"Checkpoint bloqueado: {summary['critical_blockers']}"
        )
    if data["sequences"].shape[1:] != (
        TARGET_FRAMES, EXPECTED_KEYPOINTS, EXPECTED_CHANNELS
    ):
        raise RuntimeError("Input shape inconsistente")
    if not np.isfinite(data["sequences"]).all():
        raise RuntimeError("Datos procesados no finitos")
    if split["video_id"].duplicated().any():
        raise RuntimeError("IDs duplicados en split")
    if set(split["pose_only_split"]) != {"train", "validation", "test"}:
        raise RuntimeError("Particiones incompletas")


def set_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def augment_sequence(sequence, rng):
    augmented = sequence.copy()
    if rng.random() > 0.5:
        augmented[..., 0] += rng.uniform(-0.05, 0.05)
        augmented[..., 1] += rng.uniform(-0.05, 0.05)
    if rng.random() > 0.5:
        augmented[..., :2] *= rng.uniform(0.9, 1.1)
    if rng.random() > 0.5:
        noise = rng.normal(0, 0.01, augmented[..., :2].shape)
        augmented[..., :2] += noise.astype(np.float32)
    if rng.random() > 0.5:
        angle = np.deg2rad(rng.uniform(-10.0, 10.0))
        cosine, sine = np.cos(angle), np.sin(angle)
        x = augmented[..., 0].copy()
        y = augmented[..., 1].copy()
        augmented[..., 0] = x * cosine - y * sine
        augmented[..., 1] = x * sine + y * cosine
    if rng.random() > 0.5:
        shift = int(rng.integers(-3, 4))
        if shift:
            pre_shift = augmented.copy()
            augmented = np.roll(augmented, shift, axis=0)
            if shift > 0:
                augmented[:shift] = pre_shift[0]
            else:
                augmented[shift:] = pre_shift[-1]
    return augmented.astype(np.float32)


def augment_training_data(X, y, seed):
    rng = np.random.default_rng(seed)
    augmented_X = []
    augmented_y = []
    for sequence, label in zip(X, y):
        augmented_X.append(sequence.astype(np.float32))
        augmented_y.append(int(label))
        for _ in range(AUGMENTATION_FACTOR):
            augmented_X.append(augment_sequence(sequence, rng))
            augmented_y.append(int(label))
    return (
        np.stack(augmented_X).astype(np.float32),
        np.asarray(augmented_y, dtype=np.int64),
    )


class BPFrontPoseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(EXPECTED_CHANNELS, 32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(0.40),
            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.50),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.50),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        features = self.features(inputs)
        pooled = self.pool(features)
        return self.classifier(pooled)


def make_loader(X, y, shuffle, seed):
    X_tensor = torch.from_numpy(np.asarray(X, dtype=np.float32)).permute(0, 3, 1, 2)
    y_tensor = torch.from_numpy(np.asarray(y, dtype=np.float32))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def binary_metrics(labels, predictions, probabilities):
    labels = np.asarray(labels).astype(int)
    predictions = np.asarray(predictions).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities))
        if len(np.unique(labels)) == 2 else None,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "npv": float(tn / (tn + fn)) if (tn + fn) else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    labels = []
    probabilities = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE).unsqueeze(1)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += float(loss.item()) * len(y_batch)
            total_samples += len(y_batch)
            labels.extend(y_batch.cpu().numpy().ravel().tolist())
            probabilities.extend(outputs.cpu().numpy().ravel().tolist())
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= CLASSIFICATION_THRESHOLD).astype(np.int64)
    metrics = binary_metrics(labels, predictions, probabilities)
    return total_loss / total_samples, metrics, labels, predictions, probabilities


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(y_batch)
        total_samples += len(y_batch)
    return total_loss / total_samples


def fit_model(X_train, y_train, X_validation, y_validation, seed,
              checkpoint_path, verbose_prefix=""):
    set_reproducibility(seed)
    train_loader = make_loader(X_train, y_train, shuffle=True, seed=seed)
    train_eval_loader = make_loader(X_train, y_train, shuffle=False, seed=seed)
    validation_loader = make_loader(
        X_validation, y_validation, shuffle=False, seed=seed
    )

    model = BPFrontPoseCNN().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history = []
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        optimization_loss = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        train_loss, train_metrics, _, _, _ = evaluate_model(
            model, train_eval_loader, criterion
        )
        val_loss, val_metrics, _, _, _ = evaluate_model(
            model, validation_loader, criterion
        )
        scheduler.step(val_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": epoch,
            "optimization_train_loss": optimization_loss,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": learning_rate,
        }
        for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            row[f"train_{name}"] = train_metrics[name]
            row[f"val_{name}"] = val_metrics[name]
        history.append(row)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{verbose_prefix}Epoch {epoch:3d}: "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} lr={learning_rate:.6g}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"{verbose_prefix}Early stopping en epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("No se genero checkpoint")
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch


def scale_with(scaler, X):
    return scaler.transform(X.reshape(len(X), -1)).reshape(X.shape).astype(np.float32)


def prepare_train_arrays(X_train_original, y_train, X_validation, X_test,
                         seed, scaler_path):
    scaler = StandardScaler()
    scaler.fit(X_train_original.reshape(len(X_train_original), -1))
    with open(scaler_path, "wb") as file:
        pickle.dump(scaler, file)

    X_train_augmented, y_train_augmented = augment_training_data(
        X_train_original, y_train, seed
    )
    return (
        scale_with(scaler, X_train_augmented),
        y_train_augmented,
        scale_with(scaler, X_validation),
        scale_with(scaler, X_test) if X_test is not None else None,
        scaler,
    )


def save_training_plot(history):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(history["epoch"], history["train_accuracy"], label="Train")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "bp_front_pose_training_curves.png", dpi=160)
    plt.close(fig)


def save_confusion_plot(matrix):
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Invalido (0)", "Valido (1)"])
    ax.set_yticks([0, 1], ["Invalido (0)", "Valido (1)"])
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title("BP_FRONT PoseCNN - Test")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "bp_front_pose_test_confusion_matrix.png", dpi=160)
    plt.close(fig)


def build_config(split):
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    config = {
        "experiment": EXPERIMENT_REVISION,
        "dataset_revision": "bp_352_exact_duplicate_excluded",
        "monoview": True,
        "view": "front",
        "group_id_used": False,
        "multiview_used": False,
        "seed": SEED,
        "positive_class": "1=valido",
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "pose_model": str(POSE_MODEL_PATH),
        "pose_model_sha256": file_sha256(POSE_MODEL_PATH),
        "keypoint_schema": [
            {"index": index, "name": name, "description": description}
            for index, name, description in KEYPOINT_SCHEMA
        ],
        "input_shape": [TARGET_FRAMES, EXPECTED_KEYPOINTS, EXPECTED_CHANNELS],
        "spatial_normalization": {
            "center": "median temporal del punto medio de ambos hombros",
            "scale": "mediana temporal de distancia euclidea interhombros",
            "formula": "xy_normalized=(xy-center)/scale",
        },
        "temporal_normalization": "linear resampling over normalized time to 170 frames",
        "split": {
            "strategy": "StratifiedShuffleSplit by sample",
            "train": int((split["pose_only_split"] == "train").sum()),
            "validation": int((split["pose_only_split"] == "validation").sum()),
            "test": int((split["pose_only_split"] == "test").sum()),
        },
        "scaler": "StandardScaler fit on original preprocessed train only",
        "augmentation": {
            "source": "train_only_before_scaler_transform",
            "factor": AUGMENTATION_FACTOR,
            "spatial_shift": 0.05,
            "scale": [0.9, 1.1],
            "gaussian_noise_std": 0.01,
            "rotation_degrees": 10,
            "temporal_shift_frames": 3,
            "horizontal_flip": False,
        },
        "architecture": {
            "conv_channels": [32, 64, 128],
            "global_pooling": True,
            "mlp": [128, 64, 1],
            "output": "sigmoid",
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss": "BCELoss",
            "batch_size": BATCH_SIZE,
            "epochs_max": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=5)",
        },
        "cross_validation": {
            "strategy": "StratifiedKFold",
            "folds": 5,
            "shuffle": True,
            "random_state": SEED,
            "scope": "development only; final holdout test excluded",
            "checkpoint_selection": (
                "inner stratified 15% validation inside each outer train fold"
            ),
        },
        "versions": versions,
    }
    return config


def run_training(force_test=False):
    if METRICS_PATH.exists() and not force_test:
        raise RuntimeError(
            "Ya existe una evaluacion Test. Use --force-test solo si el "
            "experimento anterior fue invalidado metodologicamente."
        )
    required = (PROCESSED_SEQUENCES_PATH, SPLIT_PATH, AUDIT_SUMMARY_PATH)
    if any(not path.exists() for path in required):
        prepare_data()
    print_safety_checkpoint()

    processed = np.load(PROCESSED_SEQUENCES_PATH, allow_pickle=True)
    ids = processed["video_ids"].astype(str)
    labels = processed["labels"].astype(int)
    sequences = processed["sequences"].astype(np.float32)
    split = pd.read_csv(SPLIT_PATH, dtype={"video_id": str})
    assignment = split.set_index("video_id")["pose_only_split"].to_dict()

    indices = {
        name: np.asarray([
            index for index, video_id in enumerate(ids)
            if assignment[video_id] == name
        ], dtype=np.int64)
        for name in ("train", "validation", "test")
    }
    train_idx = indices["train"]
    val_idx = indices["validation"]
    test_idx = indices["test"]

    X_train, y_train_aug, X_val, X_test, _ = prepare_train_arrays(
        sequences[train_idx],
        labels[train_idx],
        sequences[val_idx],
        sequences[test_idx],
        SEED,
        SCALER_PATH,
    )

    print(f"Entrenando BP_FRONT PoseCNN en {DEVICE}...")
    model, history, best_epoch = fit_model(
        X_train,
        y_train_aug,
        X_val,
        labels[val_idx],
        SEED,
        CHECKPOINT_PATH,
        verbose_prefix="[MAIN] ",
    )
    history.to_csv(HISTORY_PATH, index=False, encoding="utf-8")
    save_training_plot(history)

    criterion = nn.BCELoss()
    val_loader = make_loader(X_val, labels[val_idx], shuffle=False, seed=SEED)
    val_loss, val_metrics, _, _, _ = evaluate_model(model, val_loader, criterion)

    # Primera y unica evaluacion del holdout test en el pipeline principal.
    test_loader = make_loader(X_test, labels[test_idx], shuffle=False, seed=SEED)
    test_loss, test_metrics, test_labels, test_predictions, test_probabilities = (
        evaluate_model(model, test_loader, criterion)
    )

    prediction_rows = []
    for values in zip(
        ids[test_idx], test_labels, test_predictions, test_probabilities
    ):
        video_id, true_label, prediction, probability = values
        prediction_rows.append({
            "video_id": video_id,
            "true_label": int(true_label),
            "probability_valid": float(probability),
            "predicted_label": int(prediction),
            "correct": int(true_label) == int(prediction),
            "threshold": CLASSIFICATION_THRESHOLD,
            "positive_class": "1=valido",
        })
    pd.DataFrame(prediction_rows).to_csv(
        TEST_PREDICTIONS_PATH, index=False, encoding="utf-8"
    )

    metrics = {
        "experiment": EXPERIMENT_REVISION,
        "dataset_revision": "bp_352_exact_duplicate_excluded",
        "best_epoch": best_epoch,
        "final_epoch": int(history["epoch"].iloc[-1]),
        "validation_loss": val_loss,
        "validation": val_metrics,
        "test_loss": test_loss,
        "test": test_metrics,
        "positive_class": "1=valido",
        "threshold": CLASSIFICATION_THRESHOLD,
        "test_evaluations": 1,
    }
    save_json(METRICS_PATH, metrics)
    np.save(CONFUSION_MATRIX_PATH, np.asarray(test_metrics["confusion_matrix"]))
    save_confusion_plot(test_metrics["confusion_matrix"])
    save_json(CONFIG_PATH, build_config(split))

    print(f"Best epoch: {best_epoch}")
    print(f"Validation: {val_metrics}")
    print(f"Test: {test_metrics}")
    return metrics


def run_cross_validation():
    if not PROCESSED_SEQUENCES_PATH.exists() or not SPLIT_PATH.exists():
        prepare_data()

    processed = np.load(PROCESSED_SEQUENCES_PATH, allow_pickle=True)
    ids = processed["video_ids"].astype(str)
    labels = processed["labels"].astype(int)
    sequences = processed["sequences"].astype(np.float32)
    split = pd.read_csv(SPLIT_PATH, dtype={"video_id": str})
    heldout_test_ids = set(
        split.loc[split["pose_only_split"] == "test", "video_id"]
    )
    development_indices = np.asarray([
        index for index, video_id in enumerate(ids)
        if video_id not in heldout_test_ids
    ], dtype=np.int64)
    development_ids = ids[development_indices]
    development_labels = labels[development_indices]
    development_sequences = sequences[development_indices]

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    result_rows = []
    prediction_rows = []

    for fold, (train_local, val_local) in enumerate(
        splitter.split(development_sequences, development_labels), start=1
    ):
        fold_seed = SEED + fold
        fold_dir = CV_MODEL_DIR / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_scaler_path = fold_dir / "scaler.pkl"
        fold_checkpoint_path = fold_dir / "best_model.pth"

        inner_splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=0.15, random_state=fold_seed
        )
        fit_local_within_outer, early_local_within_outer = next(
            inner_splitter.split(
                train_local,
                development_labels[train_local],
            )
        )
        fit_local = train_local[fit_local_within_outer]
        early_local = train_local[early_local_within_outer]

        X_train, y_train_aug, X_early, X_outer_val, _ = prepare_train_arrays(
            development_sequences[fit_local],
            development_labels[fit_local],
            development_sequences[early_local],
            development_sequences[val_local],
            fold_seed,
            fold_scaler_path,
        )

        membership_rows = []
        for role, local_indices in (
            ("train_fit", fit_local),
            ("early_stopping_validation", early_local),
            ("outer_validation", val_local),
        ):
            for local_index in local_indices:
                membership_rows.append({
                    "fold": fold,
                    "video_id": development_ids[local_index],
                    "label": int(development_labels[local_index]),
                    "role": role,
                })
        pd.DataFrame(membership_rows).to_csv(
            fold_dir / "fold_membership.csv", index=False, encoding="utf-8"
        )

        print(f"\nCV fold {fold}/5 en {DEVICE}")
        model, history, best_epoch = fit_model(
            X_train,
            y_train_aug,
            X_early,
            development_labels[early_local],
            fold_seed,
            fold_checkpoint_path,
            verbose_prefix=f"[CV {fold}] ",
        )
        history.to_csv(
            fold_dir / "training_history.csv", index=False, encoding="utf-8"
        )

        loader = make_loader(
            X_outer_val,
            development_labels[val_local],
            shuffle=False,
            seed=fold_seed,
        )
        val_loss, metrics, fold_labels, fold_predictions, fold_probabilities = (
            evaluate_model(model, loader, nn.BCELoss())
        )
        row = {
            "fold": str(fold),
            "n_outer_train": int(len(train_local)),
            "n_train_original": int(len(fit_local)),
            "n_train_augmented": int(len(X_train)),
            "n_early_stopping_validation": int(len(early_local)),
            "n_outer_validation": int(len(val_local)),
            "train_invalid": int((development_labels[fit_local] == 0).sum()),
            "train_valid": int((development_labels[fit_local] == 1).sum()),
            "early_stopping_invalid": int(
                (development_labels[early_local] == 0).sum()
            ),
            "early_stopping_valid": int(
                (development_labels[early_local] == 1).sum()
            ),
            "outer_validation_invalid": int(
                (development_labels[val_local] == 0).sum()
            ),
            "outer_validation_valid": int(
                (development_labels[val_local] == 1).sum()
            ),
            "best_epoch": int(best_epoch),
            "final_epoch": int(history["epoch"].iloc[-1]),
            "outer_val_loss": float(val_loss),
        }
        row.update({key: metrics[key] for key in (
            "accuracy", "precision", "recall", "f1", "roc_auc",
            "specificity", "npv", "tn", "fp", "fn", "tp",
        )})
        result_rows.append(row)

        for values in zip(
            development_ids[val_local],
            fold_labels,
            fold_predictions,
            fold_probabilities,
        ):
            video_id, true_label, prediction, probability = values
            prediction_rows.append({
                "fold": fold,
                "video_id": video_id,
                "true_label": int(true_label),
                "probability_valid": float(probability),
                "predicted_label": int(prediction),
                "correct": int(true_label) == int(prediction),
            })
        print(f"CV fold {fold}: {metrics}")

    results = pd.DataFrame(result_rows)
    metric_names = [
        "accuracy", "precision", "recall", "f1", "roc_auc",
        "specificity", "npv",
    ]
    mean_row = {column: None for column in results.columns}
    std_row = {column: None for column in results.columns}
    mean_row["fold"] = "mean"
    std_row["fold"] = "std"
    for name in metric_names:
        mean_row[name] = float(results[name].mean())
        std_row[name] = float(results[name].std(ddof=0))
    output = pd.DataFrame(
        result_rows + [mean_row, std_row],
        columns=results.columns,
    )
    output.to_csv(CV_RESULTS_PATH, index=False, encoding="utf-8")
    pd.DataFrame(prediction_rows).to_csv(
        CV_PREDICTIONS_PATH, index=False, encoding="utf-8"
    )
    return output


def format_metric(value):
    return "N/D" if value is None else f"{float(value):.4f}"


def generate_report():
    required = (
        AUDIT_SUMMARY_PATH,
        PREPROCESSING_MANIFEST_PATH,
        SPLIT_PATH,
        HISTORY_PATH,
        METRICS_PATH,
        CV_RESULTS_PATH,
        CONFIG_PATH,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Faltan artefactos para reporte: {missing}")

    summary = json.loads(AUDIT_SUMMARY_PATH.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    split = pd.read_csv(SPLIT_PATH)
    preprocessing = pd.read_csv(PREPROCESSING_MANIFEST_PATH)
    history = pd.read_csv(HISTORY_PATH)
    cv = pd.read_csv(CV_RESULTS_PATH)
    cv_mean = cv.loc[cv["fold"] == "mean"].iloc[0]
    cv_std = cv.loc[cv["fold"] == "std"].iloc[0]
    temporal = summary["temporal_distribution"]
    test = metrics["test"]
    validation = metrics["validation"]

    split_table = []
    for name in ("train", "validation", "test"):
        subset = split.loc[split["pose_only_split"] == name]
        split_table.append(
            f"| {name.title()} | {len(subset)} | "
            f"{int((subset['label'] == 0).sum())} | "
            f"{int((subset['label'] == 1).sum())} |"
        )

    keypoint_rows = "\n".join(
        f"| {index} | `{name}` | {description} |"
        for index, name, description in KEYPOINT_SCHEMA
    )
    cv_rows = "\n".join(
        f"| {row['fold']} | {format_metric(row['accuracy'])} | "
        f"{format_metric(row['precision'])} | {format_metric(row['recall'])} | "
        f"{format_metric(row['f1'])} | {format_metric(row['roc_auc'])} |"
        for _, row in cv.loc[cv["fold"].isin(["1", "2", "3", "4", "5"])].iterrows()
    )

    report = f"""# Reporte BP_FRONT PoseCNN

## 1. Dataset utilizado

- Modalidad: Pose temporal, exclusivamente vista `front`.
- Keypoints RAW: `{KEYPOINTS_DIR}`.
- Extension: `.npy`.
- Videos originales: {summary['dataset']['videos']}.
- CSV vigente: `{LABELS_PATH}` ({summary['dataset']['csv_rows']} filas).
- Pose files: {summary['dataset']['pose_files']}, rango `{summary['dataset']['pose_id_min']}` a `{summary['dataset']['pose_id_max']}`.
- Convencion: `bp_NNN.npy`.
- No se uso `group_id`, pairing ni informacion left/right.

## 2. Modelo Pose y mapa de keypoints

- Modelo: `{summary['pose_model']['version']}`.
- SHA-256: `{summary['pose_model']['sha256']}`.
- Extractor: `{EXTRACTOR_PATH}`.
- Configuracion embebida: `kpt_shape={summary['pose_model']['kpt_shape']}`.
- Shape real uniforme: `(T, 6, 3)`.
- Features: canal 0 `x` normalizada por ancho de imagen, canal 1 `y` normalizada por alto, canal 2 `confidence` YOLO.

| index | keypoint_name | descripcion |
|---:|---|---|
{keypoint_rows}

El orden proviene de las conexiones del visualizador YOLO vigente. El comentario historico de `(0,5)` repite por error "muneca derecha"; la cadena bilateral y el hecho de que el indice 4 ya es la muneca derecha determinan que el indice 5 es la muneca izquierda. No se aplicaron flips horizontales.

## 3. Auditoria de integridad

- Layout estructural: las {summary['dataset']['pose_files']} secuencias tienen `(T,6,3)`; `T` es variable y se resume abajo.
- Dtypes: `{summary['pose_integrity']['dtype_counts']}`.
- NaN totales: {summary['pose_integrity']['total_nan']}.
- Inf totales: {summary['pose_integrity']['total_inf']}.
- Frames con missing: {summary['pose_integrity']['total_frames_with_missing']}.
- Frames completamente missing: {summary['pose_integrity']['total_completely_missing_frames']}.
- Confianza minima global: {summary['pose_integrity']['global_min_confidence']:.6f}.
- Confianza media ponderada: {summary['pose_integrity']['global_mean_confidence']:.6f}.
- Estados: `{summary['pose_integrity']['quality_counts']}`.
- Archivo por muestra: `{AUDIT_PATH}`.

### Distribucion de T_original

| N | min | P1 | P5 | P25 | median | P75 | P95 | P99 | max | mean | std |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| {temporal['N']} | {temporal['min']} | {temporal['P1']:.2f} | {temporal['P5']:.2f} | {temporal['P25']:.2f} | {temporal['median']:.2f} | {temporal['P75']:.2f} | {temporal['P95']:.2f} | {temporal['P99']:.2f} | {temporal['max']} | {temporal['mean']:.2f} | {temporal['std']:.2f} |

## 4. Quality rules y preprocessing

- `POSE_QUALITY_EXCLUDED`: shape distinto de `(T,6,3)`, menos de {MIN_FRAMES} frames, Inf, confianza fuera de `[0,1]`, mediana de confianza menor que {EXCLUDED_MEDIAN_CONFIDENCE}, ratio missing x/y mayor que {MAX_MISSING_XY_RATIO:.2%}, gap mayor que {MAX_INTERPOLATION_GAP} frames, escala interhombros degenerada o duplicado exacto no canonico.
- `POSE_QUALITY_WARNING`: missing escaso e interpolable de hasta {MAX_INTERPOLATION_GAP} frames o mediana de confianza entre {EXCLUDED_MEDIAN_CONFIDENCE} y {WARNING_MEDIAN_CONFIDENCE}.
- `POSE_QUALITY_OK`: estructura valida, finita y sin warnings.
- La tolerancia de 3 frames equivale aproximadamente a 100 ms a 30 FPS. No se copio el umbral historico de 30%; el dataset BP real tiene cero missing y permite una regla mucho mas conservadora.
- Pose utilizables: {summary['cohorts']['pose_valid']}.
- Pose excluidos: {summary['dataset']['pose_files'] - summary['cohorts']['pose_valid']}.
- Duplicados exactos detectados: `{summary['pose_integrity']['duplicate_content_groups']}`. Se conserva el ID menor como canonico; el duplicado se excluye sin borrar ni modificar RAW y sin consultar labels.
- Manifest: `{PREPROCESSING_MANIFEST_PATH}`.
- RAW originales no fueron modificados; copia derivada trazable en `{RAW_SEQUENCES_PATH}`.

## 5. Labels y clases

- Convencion confirmada: `0 = invalido`, `1 = valido`.
- Evidencia: todas las filas `resultado=valido` corresponden a 1, todas las `resultado=invalido` a 0, sin contradicciones; el loader historico solo acepta `{{0,1}}` y AGENTS.md documenta la misma convencion.
- Total: {summary['labels']['rows']}.
- Validos: {summary['labels']['valid_count']} ({summary['labels']['valid_percent']:.2f}%).
- Invalidos: {summary['labels']['invalid_count']} ({summary['labels']['invalid_percent']:.2f}%).
- Labels faltantes: {summary['labels']['missing_labels']}.
- Labels fuera de `{{0,1}}`: `{summary['labels']['labels_outside_binary']}`.
- Clase positiva de todas las metricas: `1 = valido`.

## 6. Correspondencia y cohortes

- VIDEO + CSV + POSE: {summary['correspondence']['video_csv_pose_intersection']}.
- Video sin CSV: `{summary['correspondence']['video_without_csv']}`.
- Video sin Pose: `{summary['correspondence']['video_without_pose']}`.
- CSV sin video: `{summary['correspondence']['csv_without_video']}`.
- CSV sin Pose: `{summary['correspondence']['csv_without_pose']}`.
- Pose sin video: `{summary['correspondence']['pose_without_video']}`.
- Pose sin CSV: `{summary['correspondence']['pose_without_csv']}`.
- IDs duplicados en videos/Pose/CSV: `{summary['correspondence']['duplicate_video_ids']}` / `{summary['correspondence']['duplicate_pose_ids']}` / `{summary['correspondence']['duplicate_csv_ids']}`.
- Cohorte A Pose-only: {summary['cohorts']['pose_only']}.
- Cohorte B Common Pose+Seg: {summary['cohorts']['common_pose_segmentation']}.
- Manifest segmentacion: {summary['dataset']['segmentation_manifest_rows']} registros, {summary['cohorts']['segmentation_valid']} validos y `QUALITY_ERROR={summary['cohorts']['segmentation_quality_error_ids']}`.
- `bp_349` conserva Pose valida y permanece en Pose-only; solo se excluye de common.
- `bp_352` se excluye de ambas cohortes porque su Pose y sus 110 frames decodificados son identicos a `bp_351`; `bp_351` queda como muestra canonica.

## 7. Normalizacion

### Espacial

Para cada secuencia se calcula un centro fijo:

`center = median_t((left_shoulder_t + right_shoulder_t) / 2)`

y una escala robusta fija:

`scale = median_t(||left_shoulder_t - right_shoulder_t||_2)`

Luego: `xy_normalized_t = (xy_t - center) / scale`. La confianza no cambia. Usar estadisticos fijos por secuencia reduce posicion y distancia a camara sin borrar el movimiento temporal de hombros, la simetria ni la geometria relativa de codos y munecas.

### Temporal

Se mantiene `T=170` para comparabilidad con Deadlift. Cada canal se remuestrea linealmente sobre tiempo normalizado `[0,1]`. Solo llegan a esta etapa secuencias validas y sin gaps largos; no se inventan datos para muestras excluidas.

## 8. Split monovista

- Estrategia: `StratifiedShuffleSplit` por muestra, seed {SEED}.
- Proporcion objetivo: 70/15/15.
- No se uso `GroupShuffleSplit`, `group_id` ni logica multivista.
- Split Pose-only: `{SPLIT_PATH}`.
- El mismo archivo guarda por separado `common_split`, split estratificado congelado para la futura comparacion Pose vs Segmentation vs Fusion sobre la cohorte common.

| Split | N | Invalidos (0) | Validos (1) |
|---|---:|---:|---:|
{chr(10).join(split_table)}

## 9. Scaler y augmentation

- `StandardScaler` ajustado exclusivamente con train original preprocesado, antes de augmentation.
- Validation y test solo reciben `transform`.
- Scaler: `{SCALER_PATH}`.
- Augmentation solo train: tres copias por original mas el original.
- Transformaciones independientes con probabilidad 0.5: shift x/y `+-0.05`, escala `0.9-1.1`, ruido gaussiano `sigma=0.01`, rotacion `+-10 grados`, shift temporal `+-3` frames.
- No hay horizontal flip, oversampling, undersampling ni class weights. El augmentation conserva exactamente la proporcion de clases del train.

## 10. PoseCNN e hiperparametros

- Input PyTorch: `(N,3,170,6)` desde almacenamiento `(N,170,6,3)`.
- Conv2D: 32, 64, pooling temporal, Dropout2D 0.40, Conv2D 128.
- Global average pooling.
- MLP: 128, 64, 1 con dropout 0.50 y salida Sigmoid.
- Adam, lr {LEARNING_RATE}, weight decay {WEIGHT_DECAY}, BCELoss.
- Batch {BATCH_SIZE}, maximo {MAX_EPOCHS} epochs, early stopping patience {EARLY_STOPPING_PATIENCE} por validation loss.
- Scheduler `ReduceLROnPlateau`, factor 0.5, patience 5.
- Threshold fijo 0.5; no fue optimizado con test.
- Mejor epoch: {metrics['best_epoch']}; epoch final: {metrics['final_epoch']}.
- Historial: `{HISTORY_PATH}` ({len(history)} epochs).
- Checkpoint: `{CHECKPOINT_PATH}`.

## 11. Metricas finales

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC | Specificity | NPV |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | {validation['accuracy']:.4f} | {validation['precision']:.4f} | {validation['recall']:.4f} | {validation['f1']:.4f} | {validation['roc_auc']:.4f} | {validation['specificity']:.4f} | {validation['npv']:.4f} |
| Test | {test['accuracy']:.4f} | {test['precision']:.4f} | {test['recall']:.4f} | {test['f1']:.4f} | {test['roc_auc']:.4f} | {test['specificity']:.4f} | {test['npv']:.4f} |

Test se evaluo una sola vez para la revision final deduplicada, despues de seleccionar el checkpoint por validation loss. Clase positiva: valido (1). Una ejecucion preliminar fue invalidada al descubrir el duplicado exacto `bp_351/bp_352`; sus artefactos se reemplazaron, no se usaron para ajustar hiperparametros y no forman parte de estos resultados.

### Confusion Matrix Test

| TN | FP | FN | TP |
|---:|---:|---:|---:|
| {test['tn']} | {test['fp']} | {test['fn']} | {test['tp']} |

## 12. 5-Fold Stratified Cross-Validation

- Scope: development (`train + validation`) del split principal; el holdout test final queda excluido.
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Cada fold externo empieza desde secuencias preprocesadas originales. Dentro del outer-train se reserva un 15% estratificado exclusivamente para scheduler, early stopping y seleccion de checkpoint; el scaler se ajusta solo con el inner-train original y augmentation se aplica solo al inner-train. El outer-validation se evalua una vez despues de fijar el checkpoint.
- No hay GroupKFold ni datos augmentados pregenerados compartidos.

| Fold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
{cv_rows}
| Mean | {cv_mean['accuracy']:.4f} | {cv_mean['precision']:.4f} | {cv_mean['recall']:.4f} | {cv_mean['f1']:.4f} | {cv_mean['roc_auc']:.4f} |
| Std | {cv_std['accuracy']:.4f} | {cv_std['precision']:.4f} | {cv_std['recall']:.4f} | {cv_std['f1']:.4f} | {cv_std['roc_auc']:.4f} |

## 13. Reproducibilidad

- Configuracion: `{CONFIG_PATH}`.
- Seed: {SEED}.
- Python: {config['versions']['python']}.
- PyTorch: {config['versions']['pytorch']}.
- CUDA runtime: {config['versions']['cuda_runtime']}.
- Dispositivo: {config['versions']['device']} ({config['versions']['gpu']}).
- NumPy: {config['versions']['numpy']}.
- SciPy: {config['versions']['scipy']}.
- scikit-learn: {config['versions']['scikit_learn']}.

## 14. Limitaciones y comparacion metodologica

- El esquema Pose contiene hombros, codos y munecas, pero no barra, pecho, cabeza ni gluteos. BP-02 es directamente observable; BP-01, BP-03 y BP-08 son parcial o debilmente observables. El modelo puede aprender proxies y no sustituye aun la futura fusion con segmentacion.
- El orden anatomico se reconstruye del skeleton operativo; el `data.yaml` original del entrenamiento YOLO no esta conservado.
- No se conserva el split usado para entrenar `bp_front_skeleton_v8.pt`; por ello no puede certificarse la independencia end-to-end del extractor Pose respecto a los videos del holdout.
- `T=170` sobremuestrea la mayoria de secuencias y elimina duracion absoluta, aunque mantiene comparabilidad con las PoseCNN anteriores.
- El split monovista cumple el requisito experimental actual, pero no controla identidad de atleta o sesion porque esa metadata no existe en el CSV.
- Como en Deadlift, el scaler se ajusta solo con train original y augmentation se restringe a train. A diferencia del pipeline DL lateral, no existe mapping fisico, grouping ni pairing; BP_FRONT usa estratificacion directa por muestra.
- La arquitectura conserva la familia Conv 32/64/128 + global pooling; agrega la MLP solicitada 128/64/1 sin redisenar radicalmente la baseline.

No se entreno Segmentacion CNN ni Pose+Seg Fusion y no se modifico ningun artefacto de segmentacion.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Reporte guardado: {REPORT_PATH}")


def verify_artifacts():
    required = (
        AUDIT_PATH,
        AUDIT_SUMMARY_PATH,
        RAW_SEQUENCES_PATH,
        PROCESSED_SEQUENCES_PATH,
        PREPROCESSING_MANIFEST_PATH,
        SPLIT_PATH,
        SPLIT_ARRAYS_PATH,
        SCALER_PATH,
        CONFIG_PATH,
        CHECKPOINT_PATH,
        HISTORY_PATH,
        TEST_PREDICTIONS_PATH,
        METRICS_PATH,
        CONFUSION_MATRIX_PATH,
        CV_RESULTS_PATH,
        CV_PREDICTIONS_PATH,
        REPORT_PATH,
    )
    missing = [str(path) for path in required if not path.exists()]
    assert not missing, f"Artefactos faltantes: {missing}"

    audit = pd.read_csv(AUDIT_PATH)
    preprocessing = pd.read_csv(PREPROCESSING_MANIFEST_PATH)
    split = pd.read_csv(SPLIT_PATH, dtype={"video_id": str})
    raw = np.load(RAW_SEQUENCES_PATH, allow_pickle=True)
    processed = np.load(PROCESSED_SEQUENCES_PATH, allow_pickle=True)
    arrays = np.load(SPLIT_ARRAYS_PATH)
    predictions = pd.read_csv(TEST_PREDICTIONS_PATH, dtype={"video_id": str})
    cv_results = pd.read_csv(CV_RESULTS_PATH)
    cv_predictions = pd.read_csv(CV_PREDICTIONS_PATH, dtype={"video_id": str})

    assert len(audit) == 663
    assert len(preprocessing) == 663
    assert int(preprocessing["included"].sum()) == 662
    excluded = preprocessing.loc[~preprocessing["included"]]
    assert excluded["video_id"].astype(str).tolist() == ["bp_352"]
    assert excluded["exclusion_reason"].iloc[0] == "duplicate_pose_content_of_bp_351"
    assert len(split) == 662 and split["video_id"].is_unique
    assert not any("group" in column.lower() for column in split.columns)
    assert int(split["common_member"].sum()) == 661
    assert set(split["pose_only_split"]) == {"train", "validation", "test"}
    assert set(split.loc[split["common_member"], "common_split"]) == {
        "train", "validation", "test"
    }

    raw_ids = raw["video_ids"].astype(str)
    assert len(raw_ids) == 663 and len(set(raw_ids)) == 663
    for video_id, copied_sequence in zip(raw_ids, raw["sequences"]):
        source = np.load(KEYPOINTS_DIR / f"{video_id}.npy", allow_pickle=False)
        assert np.array_equal(source, copied_sequence), (
            f"Copia RAW distinta del origen: {video_id}"
        )

    sequences = processed["sequences"]
    assert sequences.shape == (662, TARGET_FRAMES, EXPECTED_KEYPOINTS, EXPECTED_CHANNELS)
    assert np.isfinite(sequences).all()
    processed_hashes = [
        hashlib.sha256(np.ascontiguousarray(sequence).tobytes()).hexdigest()
        for sequence in sequences
    ]
    assert len(processed_hashes) == len(set(processed_hashes))

    train_original = arrays["X_train_original"]
    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)
    flattened_train = train_original.astype(np.float64).reshape(
        len(train_original), -1
    )
    assert scaler.n_features_in_ == TARGET_FRAMES * EXPECTED_KEYPOINTS * EXPECTED_CHANNELS
    assert np.allclose(scaler.mean_, flattened_train.mean(axis=0), atol=1e-7)

    train_ids = set(arrays["ids_train"].astype(str))
    validation_ids = set(arrays["ids_validation"].astype(str))
    test_ids = set(arrays["ids_test"].astype(str))
    assert not (train_ids & validation_ids)
    assert not (train_ids & test_ids)
    assert not (validation_ids & test_ids)
    assert len(predictions) == len(test_ids) == 100
    assert set(predictions["video_id"]) == test_ids

    saved_matrix = np.load(CONFUSION_MATRIX_PATH)
    recomputed_matrix = confusion_matrix(
        predictions["true_label"], predictions["predicted_label"], labels=[0, 1]
    )
    assert np.array_equal(saved_matrix, recomputed_matrix)

    development_ids = train_ids | validation_ids
    assert len(cv_predictions) == len(development_ids) == 562
    assert set(cv_predictions["video_id"]) == development_ids
    assert not (set(cv_predictions["video_id"]) & test_ids)
    assert cv_predictions["video_id"].is_unique
    assert set(cv_results["fold"].astype(str)) == {
        "1", "2", "3", "4", "5", "mean", "std"
    }

    processed_ids = processed["video_ids"].astype(str)
    processed_by_id = {
        video_id: sequences[index]
        for index, video_id in enumerate(processed_ids)
    }
    for fold in range(1, 6):
        fold_dir = CV_MODEL_DIR / f"fold_{fold}"
        membership = pd.read_csv(
            fold_dir / "fold_membership.csv", dtype={"video_id": str}
        )
        role_sets = {
            role: set(membership.loc[membership["role"] == role, "video_id"])
            for role in (
                "train_fit",
                "early_stopping_validation",
                "outer_validation",
            )
        }
        assert not (role_sets["train_fit"] & role_sets["early_stopping_validation"])
        assert not (role_sets["train_fit"] & role_sets["outer_validation"])
        assert not (
            role_sets["early_stopping_validation"]
            & role_sets["outer_validation"]
        )
        assert set().union(*role_sets.values()) == development_ids
        predicted_outer_ids = set(
            cv_predictions.loc[cv_predictions["fold"] == fold, "video_id"]
        )
        assert predicted_outer_ids == role_sets["outer_validation"]

        inner_train = np.stack([
            processed_by_id[video_id]
            for video_id in sorted(role_sets["train_fit"])
        ])
        with open(fold_dir / "scaler.pkl", "rb") as file:
            fold_scaler = pickle.load(file)
        assert np.allclose(
            fold_scaler.mean_,
            inner_train.astype(np.float64).reshape(len(inner_train), -1).mean(axis=0),
            atol=1e-7,
        )

    verification_model = BPFrontPoseCNN()
    state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    verification_model.load_state_dict(state)
    verification_model.eval()
    with torch.no_grad():
        output = verification_model(
            torch.zeros(1, EXPECTED_CHANNELS, TARGET_FRAMES, EXPECTED_KEYPOINTS)
        )
    assert output.shape == (1, 1)

    print("VERIFICACION FINAL BP_FRONT: OK")
    print("- RAW copiados sin cambios: 663/663")
    print("- Duplicado exacto bp_352 excluido; canonico bp_351 conservado")
    print("- Processed shape: (662, 170, 6, 3), NaN=0, Inf=0")
    print(
        f"- Split Pose-only: {len(train_ids)}/{len(validation_ids)}/"
        f"{len(test_ids)}, sin IDs ni contenido repetido"
    )
    print("- Common cohort: 661 con split estratificado propio")
    print("- Scaler: fit reproducido desde train original")
    print("- Test predictions: 100; confusion matrix consistente")
    print(
        "- CV: 562 development IDs una vez; inner early stopping aislado; "
        "holdout overlap=0"
    )
    print("- Checkpoint compatible con arquitectura")


def main():
    parser = argparse.ArgumentParser(description="BP_FRONT PoseCNN monovista")
    parser.add_argument(
        "--stage",
        choices=("audit", "prepare", "train", "cv", "report", "verify", "all"),
        default="all",
    )
    parser.add_argument(
        "--force-test",
        action="store_true",
        help="Permite reemplazar una evaluacion Test invalidada previamente",
    )
    args = parser.parse_args()

    if args.stage == "audit":
        run_audit()
    elif args.stage == "prepare":
        prepare_data()
    elif args.stage == "train":
        run_training(force_test=args.force_test)
    elif args.stage == "cv":
        run_cross_validation()
    elif args.stage == "report":
        generate_report()
    elif args.stage == "verify":
        verify_artifacts()
    else:
        prepare_data()
        run_training(force_test=args.force_test)
        run_cross_validation()
        generate_report()
        verify_artifacts()


if __name__ == "__main__":
    main()
