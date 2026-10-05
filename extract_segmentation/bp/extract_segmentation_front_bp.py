import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch
import ultralytics
from tqdm import tqdm
from ultralytics import YOLO

from segmentation_utils import (
    INTERACTION_FEATURE_NAMES,
    INTERACTION_PAIRS,
    OBJECT_FEATURE_NAMES,
    atomic_publish_pair,
    build_feature_arrays,
    build_interaction_features,
    build_object_features,
    build_raw_arrays,
    finite_stats,
    json_dumps,
    longest_missing_gap,
    mask_geometry,
    save_npz,
    select_main_tracks,
    sha256_file,
    summarize_interactions,
    summarize_tracks,
    track_detections,
    validate_features_file,
    validate_raw_file,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "bp" / "bp_front_seg_v2.pt"
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "bp" / "front"
DEFAULT_OUTPUT = PROJECT_ROOT / "segmentation" / "bp" / "front"
EXPECTED_CLASSES = {0: "bar", 1: "bench", 2: "person"}
PIPELINE_VERSION = "BP_FRONT_SEG_DATASET_V1"
EXTRACTOR_VERSION = "bp_front_segmentation_extractor_v1"

MANIFEST_FIELDS = [
    "video_id", "video_path", "segmentation_path", "features_path",
    "pipeline_version", "extractor_version", "git_commit",
    "frames_video", "frames_decoded", "frames_processed", "fps", "width",
    "height", "duration_seconds", "source_view", "group_id",
    "group_id_status", "paired", "status", "instances_total",
    "raw_size_bytes", "features_size_bytes", "total_size_bytes",
    "bytes_per_frame", "bytes_per_instance", "full_mask_binary_bytes",
    "cropped_mask_binary_bytes", "packed_mask_bytes",
    "packed_to_full_mask_ratio", "raw_file_to_full_mask_ratio",
    "roundtrip_masks_validated", "extraction_timestamp_utc",
]

QUALITY_FIELDS = [
    "video_id", "record_type", "class_id", "class_name", "pair_name",
    "frames_total", "frames_processed", "frames_detected", "frames_missing",
    "detection_ratio", "mean_confidence", "median_confidence",
    "min_confidence", "max_confidence", "longest_missing_gap",
    "frames_multiple_instances", "number_of_tracks", "main_track_id",
    "main_track_coverage", "minimum_distance_min_px",
    "minimum_distance_median_px", "minimum_distance_mean_px",
    "minimum_distance_max_px", "iou_min", "iou_median", "iou_mean",
    "iou_max", "overlap_frames", "overlap_ratio_when_detected",
    "proximity_frames", "proximity_ratio_when_detected",
]

ERROR_FIELDS = [
    "video_id", "video_path", "stage", "error_type", "error_message",
    "traceback", "timestamp_utc",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


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


def atomic_write_csv(path, rows, fieldnames):
    path = Path(path)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def upsert_rows(path, new_rows, fieldnames, key_fields):
    rows = read_csv_rows(path)
    new_keys = {tuple(str(row.get(field, "")) for field in key_fields) for row in new_rows}
    rows = [
        row for row in rows
        if tuple(str(row.get(field, "")) for field in key_fields) not in new_keys
    ]
    rows.extend(new_rows)
    rows.sort(key=lambda row: tuple(str(row.get(field, "")) for field in key_fields))
    atomic_write_csv(path, rows, fieldnames)


def remove_rows(path, fieldnames, predicate):
    rows = [row for row in read_csv_rows(path) if not predicate(row)]
    atomic_write_csv(path, rows, fieldnames)


def resolve_video_paths(input_dir, requested):
    if not requested:
        return sorted(input_dir.glob("*.mp4"))
    paths = []
    for value in requested:
        candidate = Path(value)
        if candidate.exists():
            paths.append(candidate.resolve())
            continue
        stem = candidate.stem
        if stem.isdigit():
            stem = "bp_%03d" % int(stem)
        if not stem.lower().startswith("bp_"):
            raise ValueError("ID de video no reconocido: %s" % value)
        path = input_dir / (stem + ".mp4")
        if not path.exists():
            raise FileNotFoundError(path)
        paths.append(path)
    unique = {path.resolve(): path.resolve() for path in paths}
    return sorted(unique.values())


def parse_result(result, frame_index, global_start, image_shape):
    if result.boxes is None or len(result.boxes) == 0:
        return []
    if result.masks is None or len(result.masks.data) != len(result.boxes):
        raise ValueError("Boxes y máscaras YOLO no tienen correspondencia 1:1")

    masks = result.masks.data.detach().cpu().numpy() > 0.5
    if masks.shape[1:] != image_shape[:2]:
        raise ValueError(
            "Máscara YOLO %s no coincide con frame %s; retina_masks debe estar activo"
            % (masks.shape[1:], image_shape[:2])
        )
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = result.boxes.conf.detach().cpu().numpy()
    polygons = result.masks.xy
    detections = []

    for instance_index, (mask, box, class_id, confidence, polygon) in enumerate(
        zip(masks, boxes, class_ids, confidences, polygons)
    ):
        class_name = str(result.names[int(class_id)])
        geometry = mask_geometry(mask)
        if geometry["area"] <= 0:
            raise ValueError("YOLO produjo una máscara vacía")
        detections.append({
            "global_index": global_start + instance_index,
            "frame_index": frame_index,
            "instance_index": instance_index,
            "class_id": int(class_id),
            "class_name": class_name,
            "confidence": float(confidence),
            "bbox_xyxy": np.asarray(box, dtype=np.float32),
            "mask": np.asarray(mask, dtype=bool),
            "polygon_xy": np.asarray(polygon, dtype=np.float32),
            "geometry": geometry,
            "track_id": -1,
        })
    return detections


def extraction_metadata(video_path, video_hash, model_path, model_hash, class_names, args, fps, width, height, frames_video):
    device_name = "CPU"
    if str(args.device).lower() != "cpu" and torch.cuda.is_available():
        device_index = int(args.device) if str(args.device).isdigit() else 0
        device_name = torch.cuda.get_device_name(device_index)
    return {
        "extraction_id": str(uuid4()),
        "extractor": "extract_segmentation_front_bp.py",
        "pipeline_version": PIPELINE_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "git_commit": args.git_commit,
        "git_worktree_dirty": args.git_worktree_dirty,
        "schema_description": "Todas las detecciones YOLO post-NMS y features de tracks principales",
        "video_id": video_path.stem,
        "video_path": str(video_path.resolve()),
        "video_sha256": video_hash,
        "video_size_bytes": int(video_path.stat().st_size),
        "video_mtime_ns": int(video_path.stat().st_mtime_ns),
        "source_view": "front",
        "group_id": video_path.stem.replace("bp_", "bp_front_individual_"),
        "group_id_status": "unverified_individual",
        "paired": 0,
        "model_path": str(model_path.resolve()),
        "model_name": model_path.name,
        "model_sha256": model_hash,
        "classes": {str(key): value for key, value in class_names.items()},
        "python_version": platform.python_version(),
        "ultralytics_version": ultralytics.__version__,
        "opencv_version": cv2.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_requested": str(args.device),
        "device_name": device_name,
        "imgsz": int(args.imgsz),
        "confidence_threshold": float(args.conf),
        "iou_threshold": float(args.iou),
        "retina_masks": True,
        "proximity_threshold_normalized": float(args.proximity_threshold),
        "proximity_semantics": "Indicador descriptivo 2D; no es contacto físico ni ground truth",
        "tracking": {
            "max_gap_frames": int(args.tracking_max_gap),
            "stationary_bench_max_gap_frames": int(args.tracking_stationary_max_gap),
            "max_assignment_cost": float(args.tracking_max_cost),
            "association": "mask IoU + bbox IoU + centroid displacement + area change",
        },
        "timestamp_source": "frame_index/fps",
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "frames_video": int(frames_video),
        "extraction_timestamp_utc": utc_now(),
        "mask_encoding": {
            "binary_threshold": 0.5,
            "crop": "minimum_nonzero_rectangle",
            "flatten_order": "C",
            "packbits_bitorder": "little",
            "archive": "np.savez_compressed",
        },
        "bar_chest_contact_generated": False,
    }


def validate_existing_configuration(metadata, video_path, video_hash, model_hash, class_names, args):
    expected = {
        "video_id": video_path.stem,
        "video_sha256": video_hash,
        "model_sha256": model_hash,
        "classes": {str(key): value for key, value in class_names.items()},
        "imgsz": int(args.imgsz),
        "confidence_threshold": float(args.conf),
        "iou_threshold": float(args.iou),
        "retina_masks": True,
        "proximity_threshold_normalized": float(args.proximity_threshold),
        "tracking": {
            "max_gap_frames": int(args.tracking_max_gap),
            "stationary_bench_max_gap_frames": int(args.tracking_stationary_max_gap),
            "max_assignment_cost": float(args.tracking_max_cost),
            "association": "mask IoU + bbox IoU + centroid displacement + area change",
        },
    }
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise ValueError(
            "Las salidas existentes usan otra configuración (%s); use --overwrite"
            % ", ".join(mismatches)
        )


def build_quality_rows(video_id, detections_by_frame, class_names, tracks, selected_tracks, interaction_features):
    frames_total = len(detections_by_frame)
    rows = []
    for class_id in sorted(class_names):
        class_name = class_names[class_id]
        detections_per_frame = [
            [detection for detection in detections if detection["class_id"] == class_id]
            for detections in detections_by_frame
        ]
        detected = np.asarray([bool(detections) for detections in detections_per_frame])
        confidences = np.asarray([
            detection["confidence"]
            for detections in detections_per_frame
            for detection in detections
        ], dtype=np.float64)
        class_tracks = [track for track in tracks.values() if track["class_id"] == class_id]
        main_track_id = selected_tracks.get(class_name, -1)
        main_track = tracks.get(main_track_id)
        confidence_stats = finite_stats(confidences)
        rows.append({
            "video_id": video_id,
            "record_type": "class",
            "class_id": class_id,
            "class_name": class_name,
            "pair_name": "",
            "frames_total": frames_total,
            "frames_processed": frames_total,
            "frames_detected": int(np.count_nonzero(detected)),
            "frames_missing": int(frames_total - np.count_nonzero(detected)),
            "detection_ratio": float(np.mean(detected)) if frames_total else 0.0,
            "mean_confidence": confidence_stats["mean"],
            "median_confidence": confidence_stats["median"],
            "min_confidence": confidence_stats["min"],
            "max_confidence": confidence_stats["max"],
            "longest_missing_gap": longest_missing_gap(detected),
            "frames_multiple_instances": int(sum(len(detections) > 1 for detections in detections_per_frame)),
            "number_of_tracks": len(class_tracks),
            "main_track_id": main_track_id,
            "main_track_coverage": len(main_track["detections"]) / frames_total if main_track and frames_total else 0.0,
        })

    distance_index = INTERACTION_FEATURE_NAMES.index("minimum_mask_distance_px")
    iou_index = INTERACTION_FEATURE_NAMES.index("iou")
    both_index = INTERACTION_FEATURE_NAMES.index("both_detected")
    overlap_index = INTERACTION_FEATURE_NAMES.index("overlap_detected")
    proximity_index = INTERACTION_FEATURE_NAMES.index("proximity_candidate")
    for pair_index, pair in enumerate(INTERACTION_PAIRS):
        pair_name = "_".join(pair)
        values = interaction_features[:, pair_index]
        both = values[:, both_index] == 1.0
        distance_stats = finite_stats(values[both, distance_index])
        iou_stats = finite_stats(values[both, iou_index])
        overlap_frames = int(np.count_nonzero(values[both, overlap_index] == 1.0))
        proximity_frames = int(np.count_nonzero(values[both, proximity_index] == 1.0))
        rows.append({
            "video_id": video_id,
            "record_type": "interaction",
            "class_id": "",
            "class_name": "",
            "pair_name": pair_name,
            "frames_total": frames_total,
            "frames_processed": frames_total,
            "frames_detected": int(np.count_nonzero(both)),
            "frames_missing": int(frames_total - np.count_nonzero(both)),
            "detection_ratio": float(np.mean(both)) if frames_total else 0.0,
            "minimum_distance_min_px": distance_stats["min"],
            "minimum_distance_median_px": distance_stats["median"],
            "minimum_distance_mean_px": distance_stats["mean"],
            "minimum_distance_max_px": distance_stats["max"],
            "iou_min": iou_stats["min"],
            "iou_median": iou_stats["median"],
            "iou_mean": iou_stats["mean"],
            "iou_max": iou_stats["max"],
            "overlap_frames": overlap_frames,
            "overlap_ratio_when_detected": overlap_frames / np.count_nonzero(both) if np.any(both) else None,
            "proximity_frames": proximity_frames,
            "proximity_ratio_when_detected": proximity_frames / np.count_nonzero(both) if np.any(both) else None,
        })
    return rows


def build_saved_report_rows(raw_path, features_path):
    raw_validation = validate_raw_file(raw_path)
    validate_features_file(features_path, raw_path)
    with np.load(raw_path, allow_pickle=False) as raw, np.load(features_path, allow_pickle=False) as features:
        metadata = raw_validation["metadata"]
        video_id = metadata["video_id"]
        frames_total = len(raw["frame_index"])
        instance_count = len(raw["instance_frame_index"])
        width = int(metadata["width"])
        height = int(metadata["height"])
        raw_size = raw_path.stat().st_size
        features_size = features_path.stat().st_size
        total_size = raw_size + features_size
        full_mask_binary_bytes = instance_count * width * height
        cropped_mask_binary_bytes = int(np.prod(raw["instance_mask_shape_hw"], axis=1).sum())
        packed_mask_bytes = int(raw["mask_bits"].nbytes)
        manifest_row = {
            "video_id": video_id,
            "video_path": metadata["video_path"],
            "segmentation_path": str(raw_path.resolve()),
            "features_path": str(features_path.resolve()),
            "pipeline_version": metadata.get("pipeline_version", PIPELINE_VERSION),
            "extractor_version": metadata.get("extractor_version", EXTRACTOR_VERSION),
            "git_commit": metadata.get("git_commit", ""),
            "frames_video": metadata["frames_video"],
            "frames_decoded": metadata["frames_decoded"],
            "frames_processed": metadata["frames_processed"],
            "fps": metadata["fps"],
            "width": width,
            "height": height,
            "duration_seconds": frames_total / float(metadata["fps"]),
            "source_view": metadata["source_view"],
            "group_id": metadata["group_id"],
            "group_id_status": metadata["group_id_status"],
            "paired": metadata["paired"],
            "status": "complete",
            "instances_total": instance_count,
            "raw_size_bytes": raw_size,
            "features_size_bytes": features_size,
            "total_size_bytes": total_size,
            "bytes_per_frame": total_size / frames_total,
            "bytes_per_instance": raw_size / instance_count if instance_count else None,
            "full_mask_binary_bytes": full_mask_binary_bytes,
            "cropped_mask_binary_bytes": cropped_mask_binary_bytes,
            "packed_mask_bytes": packed_mask_bytes,
            "packed_to_full_mask_ratio": packed_mask_bytes / full_mask_binary_bytes if full_mask_binary_bytes else None,
            "raw_file_to_full_mask_ratio": raw_size / full_mask_binary_bytes if full_mask_binary_bytes else None,
            "roundtrip_masks_validated": raw_validation["roundtrip_masks"],
            "extraction_timestamp_utc": metadata["extraction_timestamp_utc"],
        }

        track_summaries = json.loads(str(raw["track_summaries_json"]))
        selected_tracks = json.loads(str(raw["selected_tracks_json"]))
        quality_rows = []
        frame_offsets = raw["frame_instance_offsets"].astype(int)
        raw_class_ids = raw["instance_class_id"].astype(int)
        raw_confidences = raw["instance_confidence"].astype(float)
        for class_id, class_name in zip(features["class_ids"].astype(int), features["class_names"].astype(str)):
            counts = np.zeros(frames_total, dtype=np.int32)
            for frame_index in range(frames_total):
                start = frame_offsets[frame_index]
                end = frame_offsets[frame_index + 1]
                counts[frame_index] = np.count_nonzero(raw_class_ids[start:end] == class_id)
            detected = counts > 0
            class_confidences = raw_confidences[raw_class_ids == class_id]
            confidence_stats = finite_stats(class_confidences)
            class_tracks = [
                track for track in track_summaries
                if int(track["class_id"]) == int(class_id)
            ]
            main_track_id = int(selected_tracks.get(class_name, -1))
            main_track = next(
                (track for track in class_tracks if int(track["track_id"]) == main_track_id),
                None,
            )
            quality_rows.append({
                "video_id": video_id,
                "record_type": "class",
                "class_id": int(class_id),
                "class_name": class_name,
                "pair_name": "",
                "frames_total": frames_total,
                "frames_processed": frames_total,
                "frames_detected": int(np.count_nonzero(detected)),
                "frames_missing": int(frames_total - np.count_nonzero(detected)),
                "detection_ratio": float(np.mean(detected)) if frames_total else 0.0,
                "mean_confidence": confidence_stats["mean"],
                "median_confidence": confidence_stats["median"],
                "min_confidence": confidence_stats["min"],
                "max_confidence": confidence_stats["max"],
                "longest_missing_gap": longest_missing_gap(detected),
                "frames_multiple_instances": int(np.count_nonzero(counts > 1)),
                "number_of_tracks": len(class_tracks),
                "main_track_id": main_track_id,
                "main_track_coverage": (
                    int(main_track["frames_covered"]) / frames_total
                    if main_track is not None and frames_total else 0.0
                ),
            })

        interaction_features = features["interaction_features"]
        distance_index = INTERACTION_FEATURE_NAMES.index("minimum_mask_distance_px")
        iou_index = INTERACTION_FEATURE_NAMES.index("iou")
        both_index = INTERACTION_FEATURE_NAMES.index("both_detected")
        overlap_index = INTERACTION_FEATURE_NAMES.index("overlap_detected")
        proximity_index = INTERACTION_FEATURE_NAMES.index("proximity_candidate")
        for pair_index, pair_name in enumerate(features["interaction_pair_names"].astype(str)):
            values = interaction_features[:, pair_index]
            both = values[:, both_index] == 1.0
            distance_stats = finite_stats(values[both, distance_index])
            iou_stats = finite_stats(values[both, iou_index])
            overlap_frames = int(np.count_nonzero(values[both, overlap_index] == 1.0))
            proximity_frames = int(np.count_nonzero(values[both, proximity_index] == 1.0))
            both_count = int(np.count_nonzero(both))
            quality_rows.append({
                "video_id": video_id,
                "record_type": "interaction",
                "class_id": "",
                "class_name": "",
                "pair_name": pair_name,
                "frames_total": frames_total,
                "frames_processed": frames_total,
                "frames_detected": both_count,
                "frames_missing": frames_total - both_count,
                "detection_ratio": float(np.mean(both)) if frames_total else 0.0,
                "minimum_distance_min_px": distance_stats["min"],
                "minimum_distance_median_px": distance_stats["median"],
                "minimum_distance_mean_px": distance_stats["mean"],
                "minimum_distance_max_px": distance_stats["max"],
                "iou_min": iou_stats["min"],
                "iou_median": iou_stats["median"],
                "iou_mean": iou_stats["mean"],
                "iou_max": iou_stats["max"],
                "overlap_frames": overlap_frames,
                "overlap_ratio_when_detected": overlap_frames / both_count if both_count else None,
                "proximity_frames": proximity_frames,
                "proximity_ratio_when_detected": proximity_frames / both_count if both_count else None,
            })
    return manifest_row, quality_rows


def process_video(video_path, model, model_path, model_hash, class_names, output_dirs, args):
    video_id = video_path.stem
    raw_path = output_dirs["raw"] / (video_id + "_segmentation.npz")
    features_path = output_dirs["features"] / (video_id + "_features.npz")
    video_hash = sha256_file(video_path)

    if not args.overwrite and raw_path.exists() and features_path.exists():
        raw_validation = validate_raw_file(raw_path)
        validate_features_file(features_path, raw_path)
        validate_existing_configuration(
            raw_validation["metadata"], video_path, video_hash,
            model_hash, class_names, args,
        )
        print("[SKIP] %s: RAW + FEATURES existentes y válidos" % video_id)
        manifest_row, quality_rows = build_saved_report_rows(raw_path, features_path)
        return manifest_row, quality_rows, False
    if not args.overwrite and (raw_path.exists() or features_path.exists()):
        print("[RECOVER] %s: salida incompleta; se regenerará la pareja" % video_id)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError("No se pudo abrir el video")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames_video = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Metadata de video inválida")

    metadata = extraction_metadata(
        video_path, video_hash, model_path, model_hash, class_names, args,
        fps, width, height, frames_video,
    )
    detections_by_frame = []
    frame_indices = []
    timestamps = []
    frame_shapes = []
    global_instance_count = 0

    progress = tqdm(total=frames_video, desc=video_id, unit="frame", leave=True)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index = len(frame_indices)
            if frame.shape[:2] != (height, width):
                raise ValueError("La resolución cambia dentro del video")
            result = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                retina_masks=True,
                device=args.device,
                verbose=False,
            )[0]
            detections = parse_result(
                result, frame_index, global_instance_count, frame.shape
            )
            global_instance_count += len(detections)
            detections_by_frame.append(detections)
            frame_indices.append(frame_index)
            timestamps.append(frame_index / fps)
            frame_shapes.append(frame.shape[:2])
            progress.update(1)
    finally:
        progress.close()
        capture.release()

    frames_decoded = len(frame_indices)
    frames_processed = len(detections_by_frame)
    if frames_decoded != frames_processed:
        raise ValueError("frames_decoded != frames_processed")
    if frames_decoded != frames_video:
        raise ValueError(
            "La decodificación terminó en %d/%d frames" % (frames_decoded, frames_video)
        )
    if frame_indices != list(range(frames_processed)):
        raise ValueError("Índices de frames no continuos")
    if frames_processed == 0:
        raise ValueError("Video sin frames decodificados")

    tracks = track_detections(
        detections_by_frame,
        class_names,
        (height, width),
        max_gap=args.tracking_max_gap,
        max_cost=args.tracking_max_cost,
        stationary_max_gap=args.tracking_stationary_max_gap,
    )
    selected_tracks, selection_scores = select_main_tracks(
        tracks, class_names, frames_processed, (height, width)
    )
    track_summaries = summarize_tracks(tracks, selection_scores, selected_tracks)
    object_features, selected_detection_indices = build_object_features(
        detections_by_frame,
        tracks,
        selected_tracks,
        class_names,
        np.asarray(timestamps),
        (height, width),
    )
    interaction_features = build_interaction_features(
        detections_by_frame,
        selected_detection_indices,
        class_names,
        args.proximity_threshold,
    )
    interaction_summary = summarize_interactions(
        interaction_features, np.asarray(timestamps)
    )

    metadata.update({
        "frames_decoded": frames_decoded,
        "frames_processed": frames_processed,
        "instances_total": global_instance_count,
        "selected_tracks": selected_tracks,
        "roundtrip_validation": "all_instances_array_equal",
        "roundtrip_masks_expected": global_instance_count,
    })
    raw_arrays = build_raw_arrays(
        frame_indices,
        timestamps,
        frame_shapes,
        detections_by_frame,
        metadata,
        track_summaries,
        selected_tracks,
    )
    feature_arrays = build_feature_arrays(
        frame_indices,
        timestamps,
        class_names,
        selected_tracks,
        object_features,
        selected_detection_indices,
        interaction_features,
        interaction_summary,
        metadata,
    )

    raw_temp = raw_path.with_suffix(raw_path.suffix + ".tmp")
    features_temp = features_path.with_suffix(features_path.suffix + ".tmp")
    raw_temp.unlink(missing_ok=True)
    features_temp.unlink(missing_ok=True)
    try:
        save_npz(raw_temp, raw_arrays)
        save_npz(features_temp, feature_arrays)
        original_masks = [
            detection["mask"]
            for detections in detections_by_frame
            for detection in detections
        ]
        raw_validation = validate_raw_file(raw_temp, original_masks)
        validate_features_file(features_temp, raw_temp)
        atomic_publish_pair(raw_temp, raw_path, features_temp, features_path)
        validate_raw_file(raw_path, original_masks)
        validate_features_file(features_path, raw_path)
    except Exception:
        raw_temp.unlink(missing_ok=True)
        features_temp.unlink(missing_ok=True)
        raise

    full_mask_binary_bytes = global_instance_count * width * height
    cropped_mask_binary_bytes = int(sum(
        int(shape[0]) * int(shape[1])
        for shape in raw_arrays["instance_mask_shape_hw"]
    ))
    packed_mask_bytes = int(raw_arrays["mask_bits"].nbytes)
    raw_size = raw_path.stat().st_size
    features_size = features_path.stat().st_size
    total_size = raw_size + features_size
    duration = frames_processed / fps
    manifest_row = {
        "video_id": video_id,
        "video_path": str(video_path.resolve()),
        "segmentation_path": str(raw_path.resolve()),
        "features_path": str(features_path.resolve()),
        "pipeline_version": PIPELINE_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "git_commit": metadata.get("git_commit", ""),
        "frames_video": frames_video,
        "frames_decoded": frames_decoded,
        "frames_processed": frames_processed,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": duration,
        "source_view": "front",
        "group_id": metadata["group_id"],
        "group_id_status": metadata["group_id_status"],
        "paired": 0,
        "status": "complete",
        "instances_total": global_instance_count,
        "raw_size_bytes": raw_size,
        "features_size_bytes": features_size,
        "total_size_bytes": total_size,
        "bytes_per_frame": total_size / frames_processed,
        "bytes_per_instance": raw_size / global_instance_count if global_instance_count else None,
        "full_mask_binary_bytes": full_mask_binary_bytes,
        "cropped_mask_binary_bytes": cropped_mask_binary_bytes,
        "packed_mask_bytes": packed_mask_bytes,
        "packed_to_full_mask_ratio": packed_mask_bytes / full_mask_binary_bytes if full_mask_binary_bytes else None,
        "raw_file_to_full_mask_ratio": raw_size / full_mask_binary_bytes if full_mask_binary_bytes else None,
        "roundtrip_masks_validated": raw_validation["roundtrip_masks"],
        "extraction_timestamp_utc": metadata["extraction_timestamp_utc"],
    }
    quality_rows = build_quality_rows(
        video_id,
        detections_by_frame,
        class_names,
        tracks,
        selected_tracks,
        interaction_features,
    )
    print(
        "[OK] %s: %d frames, %d instancias, tracks principales %s"
        % (video_id, frames_processed, global_instance_count, selected_tracks)
    )
    return manifest_row, quality_rows, True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extrae segmentación YOLO lossless y features BP_FRONT"
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--videos", nargs="+", help="IDs o rutas de un subconjunto explícito")
    selection.add_argument(
        "--all",
        action="store_true",
        help="Procesa todo el directorio; requiere aprobación explícita",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--proximity-threshold", type=float, default=0.01)
    parser.add_argument("--tracking-max-gap", type=int, default=2)
    parser.add_argument("--tracking-stationary-max-gap", type=int, default=60)
    parser.add_argument("--tracking-max-cost", type=float, default=0.85)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.git_commit, args.git_worktree_dirty = git_state()
    numeric_arguments = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "proximity_threshold": args.proximity_threshold,
        "tracking_max_gap": args.tracking_max_gap,
        "tracking_stationary_max_gap": args.tracking_stationary_max_gap,
        "tracking_max_cost": args.tracking_max_cost,
    }
    if any(not math.isfinite(float(value)) for value in numeric_arguments.values()):
        raise ValueError("Los argumentos numéricos deben ser finitos")
    if args.imgsz <= 0 or not (0 <= args.conf <= 1) or not (0 <= args.iou <= 1):
        raise ValueError("imgsz/conf/iou fuera de rango")
    if args.proximity_threshold < 0 or args.tracking_max_gap < 0:
        raise ValueError("Umbrales de proximidad/tracking fuera de rango")
    if args.tracking_stationary_max_gap < 0 or args.tracking_max_cost < 0:
        raise ValueError("Parámetros de tracking fuera de rango")
    output_dirs = {
        "raw": args.output_dir / "raw",
        "features": args.output_dir / "features",
        "manifests": args.output_dir / "manifests",
        "previews": args.output_dir / "previews",
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dirs["manifests"] / "bp_front_segmentation_manifest.csv"
    quality_path = output_dirs["manifests"] / "bp_front_quality_report.csv"
    errors_path = output_dirs["manifests"] / "bp_front_extraction_errors.csv"
    if not errors_path.exists():
        atomic_write_csv(errors_path, [], ERROR_FIELDS)

    video_paths = resolve_video_paths(args.input_dir, None if args.all else args.videos)
    print("Videos seleccionados: %d" % len(video_paths))
    print("Modelo: %s" % args.model)
    model = YOLO(str(args.model))
    class_names = {int(key): str(value) for key, value in model.names.items()}
    print("Clases: %s" % class_names)
    if class_names != EXPECTED_CLASSES:
        raise ValueError(
            "Las clases del checkpoint cambiaron. Esperadas %s, encontradas %s"
            % (EXPECTED_CLASSES, class_names)
        )
    model_hash = sha256_file(args.model)

    completed = 0
    skipped = 0
    failed = 0
    for video_path in video_paths:
        try:
            manifest_row, quality_rows, is_new = process_video(
                video_path,
                model,
                args.model,
                model_hash,
                class_names,
                output_dirs,
                args,
            )
            upsert_rows(manifest_path, [manifest_row], MANIFEST_FIELDS, ["video_id"])
            upsert_rows(
                quality_path,
                quality_rows,
                QUALITY_FIELDS,
                ["video_id", "record_type", "class_name", "pair_name"],
            )
            remove_rows(
                errors_path,
                ERROR_FIELDS,
                lambda row, video_id=video_path.stem: row.get("video_id") == video_id,
            )
            if is_new:
                completed += 1
            else:
                skipped += 1
        except Exception as error:
            failed += 1
            error_row = {
                "video_id": video_path.stem,
                "video_path": str(video_path.resolve()),
                "stage": "extraction_or_validation",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "timestamp_utc": utc_now(),
            }
            upsert_rows(errors_path, [error_row], ERROR_FIELDS, ["video_id"])
            print("[ERROR] %s: %s" % (video_path.stem, error))

    print(
        "Extracción terminada: %d nuevos, %d omitidos válidos, %d fallidos"
        % (completed, skipped, failed)
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
