"""Cierra y audita el dataset de segmentación BP front con tracking V2.2.

Verifica RAW fuente y reconstruido, máscaras, features y cobertura de los
663 vídeos; congela configuración y genera manifiestos, previews e informe.
No ejecuta YOLO ni convierte resultados V3 en datos del clasificador.
"""

import argparse
import csv
import json
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics

from segmentation_utils import (
    INTERACTION_FEATURE_NAMES,
    longest_missing_gap,
    reconstruct_mask,
    sha256_file,
    validate_features_file,
    validate_raw_file,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATASET_VERSION = "BP_FRONT_SEG_DATASET_V1"
EXTRACTOR_VERSION = "bp_front_segmentation_extractor_v1"
TRACKING_VERSION = "bp_front_bench_tracking_v2.2"
EXPECTED_VIDEO_IDS = ["bp_%03d" % value for value in range(1, 664)]
CLASS_NAMES = ["bar", "bench", "person"]
PAIR_NAMES = ["bar_person", "person_bench", "bar_bench"]
PERCENTILES = [1, 5, 25, 50, 75, 95, 99]

DEFAULT_ROOT = PROJECT_ROOT / "segmentation" / "bp" / "front"
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "bp" / "front"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "bp" / "bp_front_seg_v2.pt"
DEFAULT_REPORT = PROJECT_ROOT / "Reporte_BP_FRONT_Segmentation_Dataset_V1.md"

MANIFEST_FIELDS = [
    "video_id", "video_path", "source_raw_path", "source_features_path",
    "segmentation_path", "features_path", "status", "quality_status",
    "quality_reasons", "source_view", "group_id", "group_id_status", "paired",
    "pipeline_version", "extractor_version", "tracking_version", "git_commit",
    "frames_video", "frames_decoded", "frames_processed", "fps", "width", "height",
    "duration_seconds", "instances_total", "raw_size_bytes", "features_size_bytes",
    "source_raw_size_bytes", "source_features_size_bytes", "total_size_bytes",
    "bytes_per_frame", "bytes_per_instance", "full_mask_binary_bytes",
    "cropped_mask_binary_bytes", "packed_mask_bytes", "packed_to_full_mask_ratio",
    "raw_file_to_full_mask_ratio", "roundtrip_masks_validated",
    "extraction_timestamp_utc", "final_validation_timestamp_utc",
]

QUALITY_FIELDS = [
    "video_id", "record_type", "class_id", "class_name", "pair_name",
    "quality_status", "frames_total", "frames_processed", "frames_detected",
    "frames_missing", "detection_ratio", "confidence_mean", "confidence_median",
    "confidence_min", "confidence_max", "longest_missing_gap", "number_of_tracks",
    "main_track_id", "main_track_coverage", "frames_multiple_instances",
    "max_instances", "tracking_warnings", "stitching_applied", "original_track_ids",
    "merged_track_id", "accepted_stitches", "accepted_bar_handoffs",
    "identity_fallback_frames",
]

ERROR_FIELDS = [
    "video_id", "video_path", "stage", "error_type", "error_message",
    "frame", "class", "instance", "reason", "traceback", "timestamp_utc",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path, text):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path, value):
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path, rows, fields):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def git_state():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def script_hashes():
    paths = [
        SCRIPT_DIR / "extract_segmentation_front_bp.py",
        SCRIPT_DIR / "segmentation_utils.py",
        SCRIPT_DIR / "tracking_front_v2" / "rebuild_tracking_front_bp_from_raw.py",
        Path(__file__).resolve(),
    ]
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def frozen_configuration(model_path):
    commit, dirty = git_state()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "pipeline_version": DATASET_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "tracking_version": TRACKING_VERSION,
        "frozen_at_utc": utc_now(),
        "git_commit": commit,
        "git_worktree_dirty": dirty,
        "script_sha256": script_hashes(),
        "model": {
            "path": str(model_path.resolve()),
            "sha256": sha256_file(model_path),
            "classes": {"0": "bar", "1": "bench", "2": "person"},
        },
        "environment": {
            "python": platform.python_version(),
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": gpu,
        },
        "inference": {
            "imgsz": 960,
            "confidence_threshold": 0.25,
            "iou_threshold": 0.7,
            "retina_masks": True,
            "proximity_threshold_normalized": 0.01,
            "proximity_semantics": "Descriptivo; no es contacto fisico ni ground truth",
        },
        "tracking_source": {
            "max_gap_frames": 2,
            "stationary_bench_max_gap_frames": 60,
            "max_assignment_cost": 0.85,
        },
        "tracking_v2": {
            "bench_anchor_weights": {
                "coverage": 0.50, "continuity": 0.15, "confidence": 0.05,
                "relative_area": 0.05, "centrality": 0.10, "stability": 0.15,
            },
            "bench_stitch": {
                "max_gap_frames": 60, "min_mask_iou": 0.80,
                "min_bbox_iou": 0.80, "max_centroid_distance_normalized": 0.01,
                "max_area_ratio": 1.25,
            },
            "bench_identity": {
                "min_shared_frames": 3, "min_mask_iou": 0.55,
                "min_containment": 0.95, "max_centroid_distance_normalized": 0.04,
                "max_area_ratio": 1.80, "min_joint_support_ratio": 0.75,
                "source_switch_penalty": 0.25, "interpolation": False,
            },
            "bar_handoff": {
                "shared_frames_min": 2, "shared_frames_max": 5,
                "min_mask_iou_each_frame": 0.90, "min_bbox_iou_each_frame": 0.65,
                "min_containment_each_frame": 0.99,
                "max_centroid_distance_normalized_each_frame": 0.01,
                "max_area_ratio_each_frame": 1.10,
            },
        },
        "prohibitions": {
            "labels_used": False,
            "temporal_interpolation": False,
            "normalization_to_170_frames": False,
            "split": False,
            "augmentation": False,
            "standard_scaler": False,
            "cnn_training": False,
        },
    }


def freeze_configuration(path, model_path):
    current = frozen_configuration(model_path)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        comparable_saved = dict(saved)
        comparable_current = dict(current)
        comparable_saved.pop("frozen_at_utc", None)
        comparable_current.pop("frozen_at_utc", None)
        if comparable_saved != comparable_current:
            raise ValueError("La configuracion congelada no coincide con codigo, modelo o entorno actuales")
        return saved
    atomic_write_json(path, current)
    return current


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None,
                "percentiles": {str(value): None for value in PERCENTILES}}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "percentiles": {str(value): float(np.percentile(values, value)) for value in PERCENTILES},
    }


def fmt(value, digits=6):
    if value is None or not np.isfinite(float(value)):
        return "N/D"
    return ("%%.%df" % digits) % float(value)


def class_quality(raw, features, audit, class_name, frames_total):
    class_id = CLASS_NAMES.index(class_name)
    offsets = raw["frame_instance_offsets"].astype(int)
    raw_class_ids = raw["instance_class_id"].astype(int)
    confidences = raw["instance_confidence"].astype(float)[raw_class_ids == class_id]
    counts = np.asarray([
        np.count_nonzero(raw_class_ids[offsets[index]:offsets[index + 1]] == class_id)
        for index in range(frames_total)
    ], dtype=int)
    detected = counts > 0
    feature_classes = features["class_names"].astype(str).tolist()
    class_position = feature_classes.index(class_name)
    object_names = features["object_feature_names"].astype(str).tolist()
    main_detected = features["object_features"][:, class_position, object_names.index("detected")] == 1.0
    selected_id = int(features["selected_track_ids"][class_position])
    tracks = audit["tracks_after_by_class"][class_name]
    merge = audit["merge_audit"]
    stitches = len(merge["accepted_stitches"]) if class_name == "bench" else 0
    handoffs = len(merge["bar_identity"]["accepted_handoffs"]) if class_name == "bar" else 0
    fallback = len(merge["bench_identity"]["fallback_frames"]) if class_name == "bench" else 0
    main_summary = next((track for track in tracks if int(track["track_id"]) == selected_id), None)
    original_ids = []
    if main_summary is not None:
        with_ids = [
            summary for summary in json.loads(str(raw["track_summaries_json"]))
            if int(summary["track_id"]) == selected_id
        ]
        if with_ids:
            original_ids = with_ids[0].get("original_track_ids", [])
    reasons = []
    if np.any(~detected):
        reasons.append("raw_missing_frames")
    if np.count_nonzero(main_detected) < np.count_nonzero(detected):
        reasons.append("main_track_below_raw_detection")
    if np.any(counts > 1):
        reasons.append("multiple_instances")
    if stitches:
        reasons.append("bench_stitching")
    if fallback:
        reasons.append("bench_identity_fallback")
    if handoffs:
        reasons.append("bar_handoff")
    confidence = stats(confidences)
    return {
        "video_id": str(raw["metadata_json"]),
        "record_type": "class", "class_id": class_id, "class_name": class_name,
        "pair_name": "", "quality_status": "QUALITY_WARNING" if reasons else "QUALITY_OK",
        "frames_total": frames_total, "frames_processed": frames_total,
        "frames_detected": int(np.count_nonzero(detected)),
        "frames_missing": int(frames_total - np.count_nonzero(detected)),
        "detection_ratio": float(np.mean(detected)), "confidence_mean": confidence["mean"],
        "confidence_median": confidence["median"], "confidence_min": confidence["min"],
        "confidence_max": confidence["max"], "longest_missing_gap": longest_missing_gap(detected),
        "number_of_tracks": len(tracks), "main_track_id": selected_id,
        "main_track_coverage": float(np.mean(main_detected)),
        "frames_multiple_instances": int(np.count_nonzero(counts > 1)),
        "max_instances": int(np.max(counts)) if counts.size else 0,
        "tracking_warnings": "|".join(reasons), "stitching_applied": int(bool(stitches or fallback or handoffs)),
        "original_track_ids": "|".join(map(str, original_ids)), "merged_track_id": selected_id,
        "accepted_stitches": stitches, "accepted_bar_handoffs": handoffs,
        "identity_fallback_frames": fallback,
        "_confidences": confidences, "_counts": counts, "_reasons": reasons,
    }


def validate_video(video_id, dataset_dir, root, configuration):
    video_path = dataset_dir / (video_id + ".mp4")
    source_raw_path = root / "raw" / (video_id + "_segmentation.npz")
    source_features_path = root / "features" / (video_id + "_features.npz")
    final_root = root / "tracking_v2"
    raw_path = final_root / "raw" / (video_id + "_segmentation.npz")
    features_path = final_root / "features" / (video_id + "_features.npz")
    audit_path = final_root / "audits" / (video_id + "_tracking_audit.json")
    for required in [video_path, source_raw_path, source_features_path, raw_path, features_path, audit_path]:
        if not required.exists():
            raise FileNotFoundError(required)
    source_validation = validate_raw_file(source_raw_path)
    validate_features_file(source_features_path, source_raw_path)
    final_validation = validate_raw_file(raw_path)
    validate_features_file(features_path, raw_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    source_hash = sha256_file(source_raw_path)
    if audit.get("source_raw_sha256") != source_hash or not audit.get("source_raw_unchanged"):
        raise ValueError("El audit no confirma integridad del RAW fuente")
    with np.load(source_raw_path, allow_pickle=False) as source, np.load(raw_path, allow_pickle=False) as raw, np.load(features_path, allow_pickle=False) as features:
        metadata = json.loads(str(raw["metadata_json"]))
        if metadata.get("model_sha256") != configuration["model"]["sha256"]:
            raise ValueError("Hash de modelo distinto al congelado")
        if metadata.get("derivation", {}).get("algorithm_version") != TRACKING_VERSION:
            raise ValueError("Version de tracking distinta a V2.2")
        frames_total = len(raw["frame_index"])
        if frames_total != int(metadata["frames_video"]):
            raise ValueError("T original no fue preservado")
        instance_count = len(raw["instance_frame_index"])
        for index in range(instance_count):
            mask = reconstruct_mask(raw, index)
            if int(np.count_nonzero(mask)) != int(raw["instance_mask_area_px"][index]):
                raise ValueError("Round-trip fallo en instance=%d" % index)
        class_rows = [class_quality(raw, features, audit, name, frames_total) for name in CLASS_NAMES]
        for row in class_rows:
            row["video_id"] = video_id
        interaction_names = features["interaction_feature_names"].astype(str).tolist()
        interaction_rows = []
        interaction_values = {}
        for pair_position, pair_name in enumerate(features["interaction_pair_names"].astype(str)):
            values = features["interaction_features"][:, pair_position]
            both = values[:, interaction_names.index("both_detected")] == 1.0
            interaction_values[pair_name] = {
                name: values[both, interaction_names.index(name)].astype(float)
                for name in INTERACTION_FEATURE_NAMES if name != "both_detected"
            }
            interaction_rows.append({
                "video_id": video_id, "record_type": "interaction", "class_id": "",
                "class_name": "", "pair_name": pair_name, "quality_status": "QUALITY_OK",
                "frames_total": frames_total, "frames_processed": frames_total,
                "frames_detected": int(np.count_nonzero(both)),
                "frames_missing": int(frames_total - np.count_nonzero(both)),
                "detection_ratio": float(np.mean(both)),
            })
        reasons = sorted({reason for row in class_rows for reason in row["_reasons"]})
        quality_status = "QUALITY_WARNING" if reasons else "QUALITY_OK"
        width = int(metadata["width"])
        height = int(metadata["height"])
        packed_bytes = int(raw["mask_bits"].nbytes)
        cropped_bytes = int(np.prod(raw["instance_mask_shape_hw"], axis=1).sum())
        full_bytes = instance_count * width * height
        raw_size = raw_path.stat().st_size
        features_size = features_path.stat().st_size
        manifest = {
            "video_id": video_id, "video_path": str(video_path.resolve()),
            "source_raw_path": str(source_raw_path.resolve()),
            "source_features_path": str(source_features_path.resolve()),
            "segmentation_path": str(raw_path.resolve()), "features_path": str(features_path.resolve()),
            "status": "WARNING" if reasons else "COMPLETE", "quality_status": quality_status,
            "quality_reasons": "|".join(reasons), "source_view": "front",
            "group_id": metadata["group_id"], "group_id_status": "unverified_individual", "paired": 0,
            "pipeline_version": DATASET_VERSION, "extractor_version": EXTRACTOR_VERSION,
            "tracking_version": TRACKING_VERSION, "git_commit": configuration.get("git_commit"),
            "frames_video": metadata["frames_video"], "frames_decoded": metadata["frames_decoded"],
            "frames_processed": metadata["frames_processed"], "fps": metadata["fps"],
            "width": width, "height": height, "duration_seconds": frames_total / float(metadata["fps"]),
            "instances_total": instance_count, "raw_size_bytes": raw_size,
            "features_size_bytes": features_size, "source_raw_size_bytes": source_raw_path.stat().st_size,
            "source_features_size_bytes": source_features_path.stat().st_size,
            "total_size_bytes": raw_size + features_size, "bytes_per_frame": (raw_size + features_size) / frames_total,
            "bytes_per_instance": raw_size / instance_count if instance_count else None,
            "full_mask_binary_bytes": full_bytes, "cropped_mask_binary_bytes": cropped_bytes,
            "packed_mask_bytes": packed_bytes,
            "packed_to_full_mask_ratio": packed_bytes / full_bytes if full_bytes else None,
            "raw_file_to_full_mask_ratio": raw_size / full_bytes if full_bytes else None,
            "roundtrip_masks_validated": final_validation["roundtrip_masks"],
            "extraction_timestamp_utc": metadata["extraction_timestamp_utc"],
            "final_validation_timestamp_utc": utc_now(),
        }
    return manifest, class_rows + interaction_rows, class_rows, interaction_values


def table_stats(title, grouped):
    lines = ["### " + title, "", "| Grupo | N | Media | Mediana | Std | Min | P1 | P5 | P25 | P50 | P75 | P95 | P99 | Max |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, values in grouped.items():
        value = stats(values)
        p = value["percentiles"]
        lines.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            name, value["count"], fmt(value["mean"]), fmt(value["median"]), fmt(value["std"]),
            fmt(value["min"]), fmt(p["1"]), fmt(p["5"]), fmt(p["25"]), fmt(p["50"]),
            fmt(p["75"]), fmt(p["95"]), fmt(p["99"]), fmt(value["max"]),
        ))
    return lines + [""]


def objectively_low_rows(rows, metric, limit):
    finite = [row for row in rows if np.isfinite(float(row[metric]))]
    if metric in {"detection_ratio", "main_track_coverage"}:
        finite = [row for row in finite if float(row[metric]) < 1.0 - 1e-12]
    elif finite and np.isclose(
        min(float(row[metric]) for row in finite),
        max(float(row[metric]) for row in finite),
    ):
        return []
    ordered = sorted(finite, key=lambda row: (float(row[metric]), row["video_id"]))
    if len(ordered) <= limit:
        return ordered
    cutoff = float(ordered[limit - 1][metric])
    return [row for row in ordered if float(row[metric]) <= cutoff + 1e-12]


def highest_rows_with_ties(rows, metric, limit):
    ordered = sorted(rows, key=lambda row: (-float(row[metric]), row["video_id"]))
    if len(ordered) <= limit:
        return ordered
    cutoff = float(ordered[limit - 1][metric])
    return [row for row in ordered if float(row[metric]) >= cutoff - 1e-12]


def worst_lines(class_rows, metric):
    rows = objectively_low_rows(class_rows, metric, 10)
    if not rows:
        return ["- Sin casos objetivamente bajos; todos los valores son perfectos o estan empatados."]
    return ["- `%s`: %s=%s" % (row["video_id"], metric, fmt(row[metric])) for row in rows]


def select_preview_cases(class_rows, interaction_by_video, audits_root):
    selected = defaultdict(set)
    for class_name in CLASS_NAMES:
        rows = [row for row in class_rows if row["class_name"] == class_name]
        for metric in ["detection_ratio", "main_track_coverage", "confidence_mean"]:
            for row in objectively_low_rows(rows, metric, 3):
                selected[row["video_id"]].add("low_%s_%s" % (class_name, metric))
        multiple = sorted(
            rows,
            key=lambda value: (-int(value["frames_multiple_instances"]), value["video_id"]),
        )
        multiple_cutoff = int(multiple[min(2, len(multiple) - 1)]["frames_multiple_instances"])
        for row in multiple:
            if int(row["frames_multiple_instances"]) < multiple_cutoff:
                break
            if int(row["frames_multiple_instances"]) > 0:
                selected[row["video_id"]].add("multiple_%s" % class_name)
        gaps = sorted(
            rows,
            key=lambda value: (-int(value["longest_missing_gap"]), value["video_id"]),
        )
        gap_cutoff = int(gaps[min(2, len(gaps) - 1)]["longest_missing_gap"])
        for row in gaps:
            if int(row["longest_missing_gap"]) < gap_cutoff:
                break
            if int(row["longest_missing_gap"]) > 0:
                selected[row["video_id"]].add("long_gap_%s" % class_name)
    stitching = []
    for audit_path in audits_root.glob("bp_*_tracking_audit.json"):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        merge = audit["merge_audit"]
        score = len(merge["accepted_stitches"]) + len(merge["bench_identity"]["fallback_frames"]) + len(merge["bar_identity"]["accepted_handoffs"])
        stitching.append({"score": score, "video_id": audit["video_id"]})
    for row in highest_rows_with_ties(stitching, "score", 5):
        if row["score"]:
            selected[row["video_id"]].add("high_stitching")
    distance_name = "minimum_mask_distance_normalized"
    for pair_name in PAIR_NAMES:
        candidates = []
        for video_id, pairs in interaction_by_video.items():
            values = pairs.get(pair_name, {}).get(distance_name, np.empty(0))
            values = values[np.isfinite(values)]
            if values.size:
                candidates.append({"distance": float(np.max(values)), "video_id": video_id})
        for row in highest_rows_with_ties(candidates, "distance", 3):
            selected[row["video_id"]].add("extreme_distance_%s" % pair_name)
    return selected


def generate_previews(selected, root):
    inspector = PROJECT_ROOT / "display" / "bp" / "inspect_segmentation_front_bp.py"
    failures = []
    for video_id in sorted(selected):
        result = subprocess.run([
            sys.executable, str(inspector), "--video", video_id,
            "--segmentation-dir", str(root / "tracking_v2"),
            "--save-previews", "--no-display", "--max-previews", "8",
        ], cwd=PROJECT_ROOT)
        if result.returncode:
            failures.append(video_id)
    return failures


def build_report(configuration, manifests, class_rows, interactions, errors, root, preview_selection):
    status_counts = defaultdict(int)
    quality_counts = defaultdict(int)
    for row in manifests:
        status_counts[row["status"]] += 1
        quality_counts[row["quality_status"]] += 1
    total_frames = sum(int(row.get("frames_processed") or 0) for row in manifests)
    total_instances = sum(int(row.get("instances_total") or 0) for row in manifests)
    roundtrip = sum(int(row.get("roundtrip_masks_validated") or 0) for row in manifests)
    detection = {name: [row["detection_ratio"] for row in class_rows if row["class_name"] == name] for name in CLASS_NAMES}
    coverage = {name: [row["main_track_coverage"] for row in class_rows if row["class_name"] == name] for name in CLASS_NAMES}
    confidence = {name: np.concatenate([row["_confidences"] for row in class_rows if row["class_name"] == name]) for name in CLASS_NAMES}
    interaction_grouped = {}
    interaction_names = [
        "minimum_mask_distance_px", "minimum_mask_distance_normalized",
        "intersection_area_px", "intersection_area_normalized", "iou",
        "a_contained_in_b_ratio", "b_contained_in_a_ratio", "proximity_candidate",
    ]
    for pair_name in PAIR_NAMES:
        for feature in interaction_names:
            arrays = [pairs[pair_name][feature] for pairs in interactions.values() if pair_name in pairs]
            interaction_grouped[pair_name + " / " + feature] = np.concatenate(arrays) if arrays else np.empty(0)
    source_raw = sum(int(row.get("source_raw_size_bytes") or 0) for row in manifests)
    source_features = sum(int(row.get("source_features_size_bytes") or 0) for row in manifests)
    final_raw = sum(int(row.get("raw_size_bytes") or 0) for row in manifests)
    final_features = sum(int(row.get("features_size_bytes") or 0) for row in manifests)
    manifest_size = sum(path.stat().st_size for path in (root / "manifests").glob("*") if path.is_file())
    full_bytes = sum(int(row.get("full_mask_binary_bytes") or 0) for row in manifests)
    packed_bytes = sum(int(row.get("packed_mask_bytes") or 0) for row in manifests)
    lines = [
        "# Reporte BP_FRONT Segmentation Dataset V1", "",
        "Fecha de cierre: `%s`" % utc_now(), "", "## 1. Configuracion exacta", "",
        "- Pipeline: `%s`." % DATASET_VERSION,
        "- Extractor: `%s`." % EXTRACTOR_VERSION,
        "- Tracking: `%s`." % TRACKING_VERSION,
        "- Modelo: `%s`." % configuration["model"]["path"],
        "- SHA-256 modelo: `%s`." % configuration["model"]["sha256"],
        "- Clases: `0=bar`, `1=bench`, `2=person`.",
        "- `imgsz=960`, `conf=0.25`, `iou=0.7`, `retina_masks=True`.",
        "- `proximity_candidate=0.01 x diagonal`, solo descriptivo.",
        "- Configuracion congelada: `segmentation/bp/front/manifests/bp_front_dataset_v1_metadata.json`.", "",
        "## 2. Entorno", "",
        "- Git: `%s` (worktree dirty al congelar: `%s`)." % (configuration.get("git_commit"), configuration.get("git_worktree_dirty")),
        "- Python: `%s`." % configuration["environment"]["python"],
        "- Ultralytics: `%s`." % configuration["environment"]["ultralytics"],
        "- PyTorch: `%s`; CUDA: `%s`." % (configuration["environment"]["torch"], configuration["environment"]["cuda"]),
        "- OpenCV: `%s`; GPU: `%s`." % (configuration["environment"]["opencv"], configuration["environment"]["gpu"]), "",
        "## 3. Integridad y alcance", "",
        "- Videos registrados: **%d/663**." % len(manifests),
        "- COMPLETE: **%d**; WARNING: **%d**; ERROR: **%d**." % (status_counts["COMPLETE"], status_counts["WARNING"], status_counts["ERROR"]),
        "- Frames procesados con T original: **%d**." % total_frames,
        "- Instancias/mascaras: **%d**." % total_instances,
        "- Round-trip correcto: **%d/%d**; fallidos: **%d**." % (roundtrip, total_instances, max(0, total_instances - roundtrip)),
        "- QUALITY_OK: **%d**; QUALITY_WARNING: **%d**; QUALITY_ERROR: **%d**." % (quality_counts["QUALITY_OK"], quality_counts["QUALITY_WARNING"], quality_counts["QUALITY_ERROR"]), "",
    ]
    lines += table_stats("Detection ratio por video", detection)
    lines += table_stats("Main track coverage por video", coverage)
    lines += table_stats("Confidence global por instancia", confidence)
    lines += ["## 4. Videos extremos", ""]
    for name in CLASS_NAMES:
        rows = [row for row in class_rows if row["class_name"] == name]
        lines += ["### 10 peores %s por detection_ratio" % name, ""] + worst_lines(rows, "detection_ratio") + [""]
        lines += ["### 10 peores %s por main_track_coverage" % name, ""] + worst_lines(rows, "main_track_coverage") + [""]
        lines += ["### 10 peores %s por confidence media" % name, ""] + worst_lines(rows, "confidence_mean") + [""]
    lines += ["## 5. Tracking, stitching y multiples instancias", ""]
    for name in CLASS_NAMES:
        rows = [row for row in class_rows if row["class_name"] == name]
        lines.append("- %s: %d videos con multiples instancias; %d frames multiples; %d maximo de instancias." % (
            name, sum(int(row["frames_multiple_instances"]) > 0 for row in rows),
            sum(int(row["frames_multiple_instances"]) for row in rows),
            max([int(row["max_instances"]) for row in rows] or [0]),
        ))
    lines += [
        "- Stitches bench aceptados: %d." % sum(int(row["accepted_stitches"]) for row in class_rows if row["class_name"] == "bench"),
        "- Frames fallback de identidad bench: %d." % sum(int(row["identity_fallback_frames"]) for row in class_rows if row["class_name"] == "bench"),
        "- Handoffs bar aceptados: %d." % sum(int(row["accepted_bar_handoffs"]) for row in class_rows if row["class_name"] == "bar"), "",
        "## 6. Interacciones globales", "",
        "`bar_person` representa barra frente a persona completa. No se infiere contacto barra-pecho.", "",
    ]
    lines += table_stats("Variables continuas y proximity_candidate", interaction_grouped)
    total_final = final_raw + final_features + manifest_size
    estimate = int(0.628 * (1024 ** 3))
    lines += [
        "## 7. Almacenamiento", "",
        "- RAW fuente: **%d bytes (%.3f GiB)**." % (source_raw, source_raw / 1024 ** 3),
        "- FEATURES fuente: **%d bytes (%.3f GiB)**." % (source_features, source_features / 1024 ** 3),
        "- RAW final V2: **%d bytes (%.3f GiB)**." % (final_raw, final_raw / 1024 ** 3),
        "- FEATURES final V2: **%d bytes (%.3f GiB)**." % (final_features, final_features / 1024 ** 3),
        "- Manifests/audits en directorio manifests: **%d bytes**." % manifest_size,
        "- Total final V2 + manifests: **%d bytes (%.3f GiB)**." % (total_final, total_final / 1024 ** 3),
        "- Bytes/frame final V2: **%s**." % fmt((final_raw + final_features) / total_frames if total_frames else None, 2),
        "- Bytes/instancia RAW final: **%s**." % fmt(final_raw / total_instances if total_instances else None, 2),
        "- Ratio packbits/full masks: **%s**." % fmt(packed_bytes / full_bytes if full_bytes else None),
        "- Diferencia frente a estimacion Smoke V1 (~0.628 GiB, RAW+FEATURES fuente): **%+.3f GiB**." % ((source_raw + source_features - estimate) / 1024 ** 3), "",
        "## 8. Previews QA", "",
        "Se seleccionaron automaticamente %d videos sin usar labels." % len(preview_selection),
    ]
    for video_id, reasons in sorted(preview_selection.items()):
        lines.append("- `%s`: %s." % (video_id, ", ".join(sorted(reasons))))
    lines += ["", "## 9. Errores", ""]
    if errors:
        for error in errors:
            lines.append("- `%s`: `%s`, %s." % (error["video_id"], error["error_type"], error["error_message"]))
    else:
        lines.append("No se registraron errores.")
    lines += [
        "", "## 10. Confirmaciones metodologicas", "",
        "- No se utilizaron labels, resultados ni codigos de falta.",
        "- No se entreno ninguna CNN ni otra red.",
        "- No hubo interpolacion ni normalizacion temporal a 170 frames.",
        "- No hubo split, augmentation ni StandardScaler.",
        "- T original se preservo y se valido para cada video completo.",
        "- Todas las instancias RAW se conservaron; el track principal sigue siendo auditable.",
        "- `proximity_candidate` no se interpreta como contacto fisico ni ground truth.",
        "- El pipeline se detiene aqui; no se inicia ninguna etapa de entrenamiento o fusion.", "",
    ]
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Congela y audita BP_FRONT_SEG_DATASET_V1")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--generate-previews", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifests_dir = args.root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    configuration_path = manifests_dir / "bp_front_dataset_v1_metadata.json"
    configuration = freeze_configuration(configuration_path, args.model)
    print("Configuracion congelada: %s" % configuration_path)
    if args.freeze_only:
        return 0

    manifests = []
    quality_rows = []
    class_rows = []
    interactions = {}
    errors = []
    for video_id in EXPECTED_VIDEO_IDS:
        video_path = args.dataset_dir / (video_id + ".mp4")
        try:
            manifest, rows, classes, pairs = validate_video(
                video_id, args.dataset_dir, args.root, configuration
            )
            manifests.append(manifest)
            quality_rows.extend(rows)
            class_rows.extend(classes)
            interactions[video_id] = pairs
            print("[OK] %s: %s" % (video_id, manifest["quality_status"]))
        except Exception as error:
            error_row = {
                "video_id": video_id, "video_path": str(video_path.resolve()),
                "stage": "final_validation", "error_type": type(error).__name__,
                "error_message": str(error), "frame": "", "class": "", "instance": "",
                "reason": str(error), "traceback": "", "timestamp_utc": utc_now(),
            }
            errors.append(error_row)
            manifests.append({
                "video_id": video_id, "video_path": str(video_path.resolve()),
                "status": "ERROR", "quality_status": "QUALITY_ERROR",
                "quality_reasons": str(error), "source_view": "front",
                "group_id": "bp_front_individual_%s" % video_id.split("_")[1],
                "group_id_status": "unverified_individual", "paired": 0,
                "pipeline_version": DATASET_VERSION, "extractor_version": EXTRACTOR_VERSION,
                "tracking_version": TRACKING_VERSION, "git_commit": configuration.get("git_commit"),
                "final_validation_timestamp_utc": utc_now(),
            })
            print("[ERROR] %s: %s" % (video_id, error))

    preview_selection = select_preview_cases(
        class_rows, interactions, args.root / "tracking_v2" / "audits"
    )
    preview_rows = [
        {"video_id": video_id, "selection_reasons": "|".join(sorted(reasons))}
        for video_id, reasons in sorted(preview_selection.items())
    ]
    atomic_write_csv(manifests_dir / "bp_front_segmentation_manifest.csv", manifests, MANIFEST_FIELDS)
    clean_quality_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in quality_rows]
    atomic_write_csv(manifests_dir / "bp_front_quality_report.csv", clean_quality_rows, QUALITY_FIELDS)
    atomic_write_csv(manifests_dir / "bp_front_extraction_errors.csv", errors, ERROR_FIELDS)
    atomic_write_csv(manifests_dir / "bp_front_qa_preview_selection.csv", preview_rows, ["video_id", "selection_reasons"])
    if args.generate_previews:
        failures = generate_previews(preview_selection, args.root)
        if failures:
            print("Previews fallidos: %s" % ", ".join(failures))
    report = build_report(
        configuration, manifests, class_rows, interactions, errors, args.root, preview_selection
    )
    atomic_write_text(args.report, report)
    print("Reporte final: %s" % args.report)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
