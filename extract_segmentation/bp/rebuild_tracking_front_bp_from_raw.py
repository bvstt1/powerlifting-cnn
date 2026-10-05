import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from segmentation_utils import (
    INTERACTION_FEATURE_NAMES,
    OBJECT_FEATURE_NAMES,
    RAW_SCHEMA_VERSION,
    RETRACKED_FEATURES_SCHEMA_VERSION,
    RETRACKED_RAW_SCHEMA_VERSION,
    atomic_publish_pair,
    bbox_iou,
    build_feature_arrays,
    build_interaction_features,
    build_object_features,
    json_dumps,
    load_detections_from_raw,
    mask_iou,
    save_npz,
    select_main_tracks,
    sha256_file,
    summarize_interactions,
    summarize_tracks,
    validate_features_file,
    validate_raw_file,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "segmentation" / "bp" / "front"
DEFAULT_OUTPUT = DEFAULT_SOURCE / "tracking_v2"
ALGORITHM_VERSION = "bp_front_bench_tracking_v2.2"
PIPELINE_VERSION = "BP_FRONT_SEG_DATASET_V1"
BENCH_IDENTITY_MIN_SHARED_FRAMES = 3

COMPARISON_FIELDS = [
    "video_id", "class_name", "frames_total", "raw_instances",
    "tracks_before", "tracks_after", "main_track_before", "main_track_after",
    "main_coverage_before", "main_coverage_after", "main_missing_before",
    "main_missing_after", "person_bench_frames_before",
    "person_bench_frames_after", "accepted_stitches", "identity_fallback_frames",
    "accepted_bar_handoffs",
    "source_raw_size_bytes", "derived_raw_size_bytes",
    "features_before_size_bytes", "features_after_size_bytes",
    "source_raw_sha256", "source_raw_unchanged", "roundtrip_masks",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def finite_summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def atomic_write_csv(path, rows, fieldnames):
    path = Path(path)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def atomic_write_json(path, data):
    path = Path(path)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    temp_path.replace(path)


def upsert_comparison(path, new_rows):
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    keys = {(row["video_id"], row["class_name"]) for row in new_rows}
    rows = [row for row in rows if (row["video_id"], row["class_name"]) not in keys]
    rows.extend(new_rows)
    rows.sort(key=lambda row: (row["video_id"], row["class_name"]))
    atomic_write_csv(path, rows, COMPARISON_FIELDS)


def tracks_from_detections(detections_by_frame):
    tracks = {}
    for detections in detections_by_frame:
        for detection in detections:
            track_id = int(detection["track_id"])
            if track_id not in tracks:
                tracks[track_id] = {
                    "track_id": track_id,
                    "class_id": int(detection["class_id"]),
                    "class_name": detection["class_name"],
                    "first_frame": int(detection["frame_index"]),
                    "last_frame": int(detection["frame_index"]),
                    "detections": [],
                }
            track = tracks[track_id]
            if track["class_id"] != detection["class_id"]:
                raise ValueError("Un track contiene clases distintas")
            track["detections"].append(detection)
            track["first_frame"] = min(track["first_frame"], detection["frame_index"])
            track["last_frame"] = max(track["last_frame"], detection["frame_index"])

    for track in tracks.values():
        track["detections"].sort(key=lambda detection: detection["frame_index"])
        frames = [detection["frame_index"] for detection in track["detections"]]
        if len(frames) != len(set(frames)):
            raise ValueError("Track %d contiene dos instancias en un frame" % track["track_id"])
    return tracks


def detection_pair_metrics(detection_a, detection_b, diagonal):
    area_a = max(int(detection_a["geometry"]["area"]), 1)
    area_b = max(int(detection_b["geometry"]["area"]), 1)
    intersection = int(np.count_nonzero(detection_a["mask"] & detection_b["mask"]))
    center_a = np.array([
        detection_a["geometry"]["centroid_x"],
        detection_a["geometry"]["centroid_y"],
    ])
    center_b = np.array([
        detection_b["geometry"]["centroid_x"],
        detection_b["geometry"]["centroid_y"],
    ])
    return {
        "mask_iou": mask_iou(detection_a["mask"], detection_b["mask"]),
        "bbox_iou": bbox_iou(detection_a["bbox_xyxy"], detection_b["bbox_xyxy"]),
        "centroid_distance_normalized": float(np.linalg.norm(center_a - center_b) / diagonal),
        "area_ratio": float(max(area_a, area_b) / min(area_a, area_b)),
        "smaller_containment_ratio": float(intersection / min(area_a, area_b)),
    }


def aggregate_pair_metrics(pairs, diagonal):
    metrics = [detection_pair_metrics(a, b, diagonal) for a, b in pairs]
    result = {
        name: float(np.median([metric[name] for metric in metrics]))
        for name in metrics[0]
    }
    result["pair_count"] = len(metrics)
    result["metric_summaries"] = {
        name: finite_summary([metric[name] for metric in metrics])
        for name in metrics[0]
    }
    result["nested_support_ratio"] = float(np.mean([
        metric["mask_iou"] >= 0.55
        and metric["smaller_containment_ratio"] >= 0.95
        and metric["centroid_distance_normalized"] <= 0.04
        and metric["area_ratio"] <= 1.80
        for metric in metrics
    ]))
    result["duplicate_support_ratio"] = float(np.mean([
        metric["mask_iou"] >= 0.85
        and metric["bbox_iou"] >= 0.75
        and metric["centroid_distance_normalized"] <= 0.01
        and metric["area_ratio"] <= 1.25
        for metric in metrics
    ]))
    return result


def compare_tracks(track_a, track_b, diagonal, endpoint_count=3):
    by_frame_a = {detection["frame_index"]: detection for detection in track_a["detections"]}
    by_frame_b = {detection["frame_index"]: detection for detection in track_b["detections"]}
    shared_frames = sorted(set(by_frame_a) & set(by_frame_b))
    if shared_frames:
        pairs = [(by_frame_a[frame], by_frame_b[frame]) for frame in shared_frames]
        mode = "temporal_overlap"
        missing_gap = -1
    elif track_a["last_frame"] < track_b["first_frame"]:
        tail = track_a["detections"][-endpoint_count:]
        head = track_b["detections"][:endpoint_count]
        pairs = [(a, b) for a in tail for b in head]
        mode = "forward_gap"
        missing_gap = int(track_b["first_frame"] - track_a["last_frame"] - 1)
    elif track_b["last_frame"] < track_a["first_frame"]:
        tail = track_b["detections"][-endpoint_count:]
        head = track_a["detections"][:endpoint_count]
        pairs = [(a, b) for a in tail for b in head]
        mode = "reverse_gap"
        missing_gap = int(track_a["first_frame"] - track_b["last_frame"] - 1)
    else:
        raise ValueError("Intervalos de track inconsistentes")
    result = aggregate_pair_metrics(pairs, diagonal)
    result.update({
        "comparison_mode": mode,
        "shared_frames": shared_frames,
        "shared_frame_count": len(shared_frames),
        "missing_gap_frames": missing_gap,
    })
    return result


def stitch_non_overlapping_bench_tracks(tracks, image_shape_hw):
    height, width = map(int, image_shape_hw)
    diagonal = math.hypot(width, height)
    bench_tracks = sorted(
        [track for track in tracks.values() if track["class_name"] == "bench"],
        key=lambda track: (track["first_frame"], track["track_id"]),
    )
    candidates = []
    rejected = []
    for track_a in bench_tracks:
        for track_b in bench_tracks:
            if track_a["last_frame"] >= track_b["first_frame"]:
                continue
            metrics = compare_tracks(track_a, track_b, diagonal)
            accepted = (
                0 <= metrics["missing_gap_frames"] <= 60
                and metrics["mask_iou"] >= 0.80
                and metrics["bbox_iou"] >= 0.80
                and metrics["centroid_distance_normalized"] <= 0.01
                and metrics["area_ratio"] <= 1.25
            )
            cost = (
                0.50 * (1.0 - metrics["mask_iou"])
                + 0.20 * (1.0 - metrics["bbox_iou"])
                + 0.20 * min(metrics["centroid_distance_normalized"] / 0.01, 1.0)
                + 0.10 * min(math.log(metrics["area_ratio"]) / math.log(1.25), 1.0)
                + 0.05 * metrics["missing_gap_frames"] / 60.0
            )
            record = {
                "from_track_id": int(track_a["track_id"]),
                "to_track_id": int(track_b["track_id"]),
                "from_last_frame": int(track_a["last_frame"]),
                "to_first_frame": int(track_b["first_frame"]),
                "cost": float(cost),
                **metrics,
            }
            if accepted:
                candidates.append(record)
            else:
                record["decision"] = "rejected_geometry_or_gap"
                rejected.append(record)

    parent = {track["track_id"]: track["track_id"] for track in bench_tracks}

    def root(track_id):
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    used_predecessors = set()
    used_successors = set()
    accepted = []
    for record in sorted(candidates, key=lambda value: (value["cost"], value["from_track_id"], value["to_track_id"])):
        source = record["from_track_id"]
        target = record["to_track_id"]
        if source in used_predecessors or target in used_successors:
            record["decision"] = "rejected_assignment_conflict"
            rejected.append(record)
            continue
        source_root = root(source)
        target_root = root(target)
        if source_root == target_root:
            continue
        parent[target_root] = source_root
        used_predecessors.add(source)
        used_successors.add(target)
        record["decision"] = "accepted_stitch"
        accepted.append(record)

    mapping = {track_id: root(track_id) for track_id in parent}
    return mapping, accepted, rejected


def identity_relation(metrics):
    duplicate = (
        metrics["mask_iou"] >= 0.85
        and metrics["bbox_iou"] >= 0.75
        and metrics["centroid_distance_normalized"] <= 0.01
        and metrics["area_ratio"] <= 1.25
        and metrics["duplicate_support_ratio"] >= 0.75
    )
    nested = (
        metrics["mask_iou"] >= 0.55
        and metrics["smaller_containment_ratio"] >= 0.95
        and metrics["centroid_distance_normalized"] <= 0.04
        and metrics["area_ratio"] <= 1.80
        and metrics["nested_support_ratio"] >= 0.75
    )
    if duplicate:
        return "duplicate_hypothesis"
    if nested:
        return "nested_same_physical_bench"
    return None


def consolidate_bar_handoff(tracks, main_track_id, image_shape_hw):
    height, width = map(int, image_shape_hw)
    diagonal = math.hypot(width, height)
    if main_track_id not in tracks or tracks[main_track_id]["class_name"] != "bar":
        raise ValueError("El track bar principal no existe")

    selected_tracks = [tracks[main_track_id]]
    accepted = []
    rejected = []
    used_ids = {main_track_id}
    while True:
        source = selected_tracks[-1]
        candidates = []
        for target in sorted(tracks.values(), key=lambda value: value["track_id"]):
            if target["class_name"] != "bar" or target["track_id"] in used_ids:
                continue
            overlap_start = max(source["first_frame"], target["first_frame"])
            overlap_end = min(source["last_frame"], target["last_frame"])
            shared_frames = max(0, overlap_end - overlap_start + 1)
            if not (
                source["first_frame"] < target["first_frame"]
                and target["first_frame"] <= source["last_frame"]
                and source["last_frame"] < target["last_frame"]
                and 2 <= shared_frames <= 5
            ):
                continue

            metrics = compare_tracks(source, target, diagonal)
            summaries = metrics["metric_summaries"]
            accepted_geometry = (
                summaries["mask_iou"]["min"] >= 0.90
                and summaries["bbox_iou"]["min"] >= 0.65
                and summaries["smaller_containment_ratio"]["min"] >= 0.99
                and summaries["centroid_distance_normalized"]["max"] <= 0.01
                and summaries["area_ratio"]["max"] <= 1.10
            )
            source_last = source["detections"][-1]
            target_after = next((
                detection for detection in target["detections"]
                if detection["frame_index"] == source["last_frame"] + 1
            ), None)
            transition = (
                detection_pair_metrics(source_last, target_after, diagonal)
                if target_after is not None else None
            )
            accepted_transition = transition is not None and (
                transition["mask_iou"] >= 0.50
                and transition["bbox_iou"] >= 0.50
                and transition["centroid_distance_normalized"] <= 0.02
                and transition["area_ratio"] <= 1.25
            )
            cost = (
                0.55 * (1.0 - metrics["mask_iou"])
                + 0.20 * (1.0 - metrics["smaller_containment_ratio"])
                + 0.15 * min(metrics["centroid_distance_normalized"] / 0.01, 1.0)
                + 0.10 * min((metrics["area_ratio"] - 1.0) / 0.10, 1.0)
            )
            record = {
                "from_track_id": int(source["track_id"]),
                "to_track_id": int(target["track_id"]),
                "from_last_frame": int(source["last_frame"]),
                "to_first_frame": int(target["first_frame"]),
                "cost": float(cost),
                "transition_metrics": transition,
                **metrics,
            }
            if accepted_geometry and accepted_transition:
                candidates.append((cost, int(target["track_id"]), target, record))
            else:
                record["decision"] = "rejected_bar_handoff_geometry"
                rejected.append(record)

        if not candidates:
            break
        _, _, target, record = min(candidates, key=lambda value: (value[0], value[1]))
        record["decision"] = "accepted_bar_handoff"
        accepted.append(record)
        selected_tracks.append(target)
        used_ids.add(int(target["track_id"]))

    selected_by_frame = {}
    for position, track in enumerate(selected_tracks):
        previous_last = (
            selected_tracks[position - 1]["last_frame"] if position else -1
        )
        for detection in track["detections"]:
            if detection["frame_index"] > previous_last:
                selected_by_frame[detection["frame_index"]] = detection
    return selected_by_frame, {
        "anchor_track_id": int(main_track_id),
        "identity_track_ids": [int(track["track_id"]) for track in selected_tracks],
        "accepted_handoffs": accepted,
        "rejected_handoffs": rejected,
        "criterion": (
            "2-5 shared frames; per-frame mask IoU>=0.90, bbox IoU>=0.65, "
            "containment>=0.99, centroid<=0.01 diagonal, area ratio<=1.10; "
            "immediate successor transition required"
        ),
    }


def select_bench_anchor(tracks, frames_total, image_shape_hw):
    height, width = map(int, image_shape_hw)
    diagonal = math.hypot(width, height)
    image_area = height * width
    bench_tracks = [
        track for track in tracks.values() if track["class_name"] == "bench"
    ]
    if not bench_tracks:
        raise ValueError("No existen tracks bench")

    max_median_area = max(
        float(np.median([
            detection["geometry"]["area"] for detection in track["detections"]
        ]))
        for track in bench_tracks
    )
    scores = {}
    for track in bench_tracks:
        detections = track["detections"]
        frames = np.asarray([
            detection["frame_index"] for detection in detections
        ], dtype=np.int32)
        centers = np.asarray([
            [
                detection["geometry"]["centroid_x"],
                detection["geometry"]["centroid_y"],
            ]
            for detection in detections
        ], dtype=np.float64)
        areas = np.asarray([
            detection["geometry"]["area"] for detection in detections
        ], dtype=np.float64)
        confidences = np.asarray([
            detection["confidence"] for detection in detections
        ], dtype=np.float64)
        coverage = len(np.unique(frames)) / max(frames_total, 1)
        span = int(frames.max() - frames.min() + 1)
        continuity = len(np.unique(frames)) / max(span, 1)
        if len(centers) > 1:
            displacements = np.linalg.norm(np.diff(centers, axis=0), axis=1)
            frame_steps = np.maximum(np.diff(frames), 1)
            displacement_per_frame = displacements / frame_steps
            stability = 1.0 - min(
                float(np.median(displacement_per_frame)) / (0.10 * diagonal), 1.0
            )
        else:
            stability = 0.0
        image_center = np.asarray([width / 2.0, height / 2.0])
        centrality = 1.0 - min(
            float(np.median(np.linalg.norm(centers - image_center, axis=1)))
            / (0.5 * diagonal),
            1.0,
        )
        median_area = float(np.median(areas))
        relative_area = median_area / max_median_area if max_median_area else 0.0
        components = {
            "coverage": float(coverage),
            "continuity": float(continuity),
            "mean_confidence": float(np.mean(confidences)),
            "median_area_normalized": median_area / image_area,
            "relative_area": float(relative_area),
            "centrality": float(centrality),
            "stability_per_frame": float(stability),
        }
        components["anchor_score"] = float(
            0.50 * coverage
            + 0.15 * continuity
            + 0.05 * components["mean_confidence"]
            + 0.05 * relative_area
            + 0.10 * centrality
            + 0.15 * stability
        )
        scores[int(track["track_id"])] = components

    selected = max(
        bench_tracks,
        key=lambda track: (
            scores[int(track["track_id"])]["anchor_score"],
            scores[int(track["track_id"])]["coverage"],
            -int(track["track_id"]),
        ),
    )
    return int(selected["track_id"]), scores


def consolidate_bench_identity(detections_by_frame, tracks, anchor_track_id, image_shape_hw):
    height, width = map(int, image_shape_hw)
    diagonal = math.hypot(width, height)
    if anchor_track_id not in tracks:
        raise ValueError("El track bench ancla no existe")
    anchor = tracks[anchor_track_id]
    related = {anchor["track_id"]: {"relation": "anchor", "metrics": None}}
    rejected = []

    for track in sorted(tracks.values(), key=lambda value: value["track_id"]):
        if track["class_name"] != "bench" or track["track_id"] == anchor["track_id"]:
            continue
        metrics = compare_tracks(anchor, track, diagonal)
        relation = (
            identity_relation(metrics)
            if metrics["shared_frame_count"] >= BENCH_IDENTITY_MIN_SHARED_FRAMES
            else None
        )
        record = {
            "anchor_track_id": int(anchor["track_id"]),
            "candidate_track_id": int(track["track_id"]),
            "relation": relation,
            **metrics,
        }
        if relation:
            related[track["track_id"]] = {"relation": relation, "metrics": metrics}
        else:
            record["decision"] = "rejected_not_same_identity"
            rejected.append(record)

    def candidate_score(detection):
        relation_data = related[detection["track_id"]]
        metrics = relation_data["metrics"]
        if metrics is None:
            return 1.0 + 0.05 * detection["confidence"]
        return float(
            0.40 * metrics["mask_iou"]
            + 0.25 * metrics["smaller_containment_ratio"]
            + 0.15 * metrics["bbox_iou"]
            + 0.10 * (1.0 - min(metrics["centroid_distance_normalized"] / 0.04, 1.0))
            + 0.05 * (1.0 - min(math.log(metrics["area_ratio"]) / math.log(1.80), 1.0))
            + 0.05 * detection["confidence"]
        )

    candidates_by_frame = {}
    for frame_index, frame_detections in enumerate(detections_by_frame):
        candidates = sorted(
            [
                detection for detection in frame_detections
                if detection["class_name"] == "bench" and detection["track_id"] in related
            ],
            key=lambda detection: detection["global_index"],
        )
        if candidates:
            candidates_by_frame[frame_index] = candidates

    switch_penalty = 0.25
    dynamic = {}
    backpointers = {}
    previous_frame = None
    for frame_index in sorted(candidates_by_frame):
        candidates = candidates_by_frame[frame_index]
        dynamic[frame_index] = {}
        backpointers[frame_index] = {}
        for detection in candidates:
            instance_index = detection["global_index"]
            emission = candidate_score(detection)
            if previous_frame is None or previous_frame != frame_index - 1:
                dynamic[frame_index][instance_index] = emission
                backpointers[frame_index][instance_index] = None
                continue
            choices = []
            for previous_index, previous_score in dynamic[previous_frame].items():
                previous_detection = next(
                    value for value in candidates_by_frame[previous_frame]
                    if value["global_index"] == previous_index
                )
                penalty = (
                    switch_penalty
                    if previous_detection["track_id"] != detection["track_id"]
                    else 0.0
                )
                choices.append((
                    previous_score + emission - penalty,
                    -previous_detection["global_index"],
                    previous_index,
                ))
            best = max(choices)
            dynamic[frame_index][instance_index] = best[0]
            backpointers[frame_index][instance_index] = best[2]
        previous_frame = frame_index

    selected_by_frame = {}
    contiguous_runs = []
    current_run = []
    previous = None
    for frame_index in sorted(candidates_by_frame):
        if previous is None or frame_index == previous + 1:
            current_run.append(frame_index)
        else:
            contiguous_runs.append(current_run)
            current_run = [frame_index]
        previous = frame_index
    if current_run:
        contiguous_runs.append(current_run)

    by_global_index = {
        detection["global_index"]: detection
        for candidates in candidates_by_frame.values() for detection in candidates
    }
    for run in contiguous_runs:
        last_frame = run[-1]
        last_index = max(
            dynamic[last_frame],
            key=lambda index: (dynamic[last_frame][index], -index),
        )
        for frame_index in reversed(run):
            selected_by_frame[frame_index] = by_global_index[last_index]
            last_index = backpointers[frame_index][last_index]

    fallback_records = []
    for frame_index, selected in sorted(selected_by_frame.items()):
        if selected["track_id"] == anchor["track_id"]:
            continue
        fallback_records.append({
            "frame_index": frame_index,
            "instance_index": int(selected["global_index"]),
            "original_track_id": int(selected["original_track_id"]),
            "source_track_id": int(selected["track_id"]),
            "relation": related[selected["track_id"]]["relation"],
            "selection_score": candidate_score(selected),
        })

    identity_audit = {
        "anchor_track_id": int(anchor["track_id"]),
        "identity_track_ids": sorted(int(track_id) for track_id in related),
        "relations": [
            {
                "track_id": int(track_id),
                "relation": data["relation"],
                "metrics": data["metrics"],
            }
            for track_id, data in sorted(related.items())
        ],
        "fallback_frames": fallback_records,
        "framewise_selection": "dynamic programming with source-track switch penalty",
        "source_track_switch_penalty": switch_penalty,
        "minimum_shared_frames": BENCH_IDENTITY_MIN_SHARED_FRAMES,
        "rejected_relations": rejected,
    }
    return selected_by_frame, identity_audit


def recalculate_temporal_features(object_features, selected_indices, detections_by_index, timestamps, image_shape_hw):
    names = {name: index for index, name in enumerate(OBJECT_FEATURE_NAMES)}
    temporal_start = names["dt_seconds"]
    object_features[:, :, temporal_start:] = np.nan
    height, width = map(int, image_shape_hw)

    for class_position in range(object_features.shape[1]):
        for frame_index in range(1, object_features.shape[0]):
            current_index = int(selected_indices[frame_index, class_position])
            previous_index = int(selected_indices[frame_index - 1, class_position])
            if current_index < 0 or previous_index < 0:
                continue
            current = detections_by_index[current_index]
            previous = detections_by_index[previous_index]
            if current["original_track_id"] != previous["original_track_id"]:
                continue
            dt = float(timestamps[frame_index] - timestamps[frame_index - 1])
            if not np.isfinite(dt) or dt <= 0:
                continue
            dx = current["geometry"]["centroid_x"] - previous["geometry"]["centroid_x"]
            dy = current["geometry"]["centroid_y"] - previous["geometry"]["centroid_y"]
            vx = dx / dt
            vy = dy / dt
            values = {
                "dt_seconds": dt,
                "centroid_dx_px": dx,
                "centroid_dy_px": dy,
                "centroid_dx_normalized": dx / width,
                "centroid_dy_normalized": dy / height,
                "centroid_vx_px_s": vx,
                "centroid_vy_px_s": vy,
                "centroid_vx_normalized_s": (dx / width) / dt,
                "centroid_vy_normalized_s": (dy / height) / dt,
            }
            for name, value in values.items():
                object_features[frame_index, class_position, names[name]] = value

            if frame_index < 2:
                continue
            older_index = int(selected_indices[frame_index - 2, class_position])
            if older_index < 0:
                continue
            older = detections_by_index[older_index]
            if older["original_track_id"] != current["original_track_id"]:
                continue
            previous_vx = object_features[frame_index - 1, class_position, names["centroid_vx_px_s"]]
            previous_vy = object_features[frame_index - 1, class_position, names["centroid_vy_px_s"]]
            if not np.all(np.isfinite([previous_vx, previous_vy])):
                continue
            accelerations = {
                "centroid_ax_px_s2": (vx - previous_vx) / dt,
                "centroid_ay_px_s2": (vy - previous_vy) / dt,
                "centroid_ax_normalized_s2": ((vx - previous_vx) / width) / dt,
                "centroid_ay_normalized_s2": ((vy - previous_vy) / height) / dt,
            }
            for name, value in accelerations.items():
                object_features[frame_index, class_position, names[name]] = value


def track_detail(track, score_components, frames_total):
    detections = track["detections"]
    frames = np.asarray([detection["frame_index"] for detection in detections], dtype=int)
    confidences = [detection["confidence"] for detection in detections]
    areas = [detection["geometry"]["area"] for detection in detections]
    centroids = np.asarray([
        [detection["geometry"]["centroid_x"], detection["geometry"]["centroid_y"]]
        for detection in detections
    ])
    boxes = np.asarray([detection["bbox_xyxy"] for detection in detections])
    temporal_mask_ious = []
    temporal_bbox_ious = []
    for previous, current in zip(detections, detections[1:]):
        if current["frame_index"] == previous["frame_index"] + 1:
            temporal_mask_ious.append(mask_iou(previous["mask"], current["mask"]))
            temporal_bbox_ious.append(bbox_iou(previous["bbox_xyxy"], current["bbox_xyxy"]))
    gaps = []
    for previous, current in zip(frames, frames[1:]):
        if current > previous + 1:
            gaps.append({"start": int(previous + 1), "end": int(current - 1), "length": int(current - previous - 1)})
    return {
        "track_id": int(track["track_id"]),
        "class_name": track["class_name"],
        "first_frame": int(frames.min()),
        "last_frame": int(frames.max()),
        "frames": int(len(frames)),
        "coverage": float(len(frames) / frames_total),
        "frame_indices": frames.tolist(),
        "gaps": gaps,
        "confidence": finite_summary(confidences),
        "area_px": finite_summary(areas),
        "centroid_x_px": finite_summary(centroids[:, 0]),
        "centroid_y_px": finite_summary(centroids[:, 1]),
        "bbox_x1_px": finite_summary(boxes[:, 0]),
        "bbox_y1_px": finite_summary(boxes[:, 1]),
        "bbox_x2_px": finite_summary(boxes[:, 2]),
        "bbox_y2_px": finite_summary(boxes[:, 3]),
        "temporal_mask_iou": finite_summary(temporal_mask_ious),
        "temporal_bbox_iou": finite_summary(temporal_bbox_ious),
        "score_components": score_components,
    }


def source_preserved(source, derived):
    immutable_arrays = [
        name for name in source.files
        if name not in {
            "schema_version", "metadata_json", "track_summaries_json",
            "selected_tracks_json", "instance_track_id",
        }
    ]
    for name in immutable_arrays:
        supports_nan = source[name].dtype.kind in {"f", "c"}
        if not np.array_equal(source[name], derived[name], equal_nan=supports_nan):
            raise ValueError("El array RAW inmutable cambió: %s" % name)
    if not np.array_equal(source["instance_track_id"], derived["instance_original_track_id"]):
        raise ValueError("No se conservaron los track IDs originales")
    return immutable_arrays


def existing_output_matches_source(output_raw_path, output_audit_path, source_hash):
    try:
        with np.load(output_raw_path, allow_pickle=False) as output:
            metadata = json.loads(str(output["metadata_json"]))
        with output_audit_path.open("r", encoding="utf-8") as file:
            audit = json.load(file)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    derivation = metadata.get("derivation", {})
    return (
        derivation.get("algorithm_version") == ALGORITHM_VERSION
        and derivation.get("source_raw_sha256") == source_hash
        and audit.get("source_raw_sha256") == source_hash
        and audit.get("merge_audit", {}).get("algorithm_version") == ALGORITHM_VERSION
    )


def process_video(video_id, source_root, output_root, overwrite):
    source_raw_path = source_root / "raw" / (video_id + "_segmentation.npz")
    source_features_path = source_root / "features" / (video_id + "_features.npz")
    output_raw_path = output_root / "raw" / (video_id + "_segmentation.npz")
    output_features_path = output_root / "features" / (video_id + "_features.npz")
    output_audit_path = output_root / "audits" / (video_id + "_tracking_audit.json")

    if not source_raw_path.exists() or not source_features_path.exists():
        raise FileNotFoundError("Falta RAW/features fuente para %s" % video_id)
    if source_root.resolve() == output_root.resolve():
        raise ValueError("source-dir y output-dir deben ser distintos")
    source_hash_before = sha256_file(source_raw_path)
    if not overwrite and output_raw_path.exists() and output_features_path.exists() and output_audit_path.exists():
        validate_raw_file(output_raw_path)
        validate_features_file(output_features_path, output_raw_path)
        if existing_output_matches_source(
            output_raw_path, output_audit_path, source_hash_before
        ):
            print("[SKIP] %s tracking V2 existente y vigente" % video_id)
            return None
        print("[REBUILD] %s tracking V2 obsoleto" % video_id)

    source_validation = validate_raw_file(source_raw_path)
    if source_validation["schema_version"] != RAW_SCHEMA_VERSION:
        raise ValueError(
            "El rebuild requiere RAW original %s; recibido %s"
            % (RAW_SCHEMA_VERSION, source_validation["schema_version"])
        )
    validate_features_file(source_features_path, source_raw_path)

    with np.load(source_raw_path, allow_pickle=False) as source, np.load(source_features_path, allow_pickle=False) as old_features:
        source_arrays = {name: source[name].copy() for name in source.files}
        metadata = json.loads(str(source["metadata_json"]))
        original_summaries = json.loads(str(source["track_summaries_json"]))
        original_selected = json.loads(str(source["selected_tracks_json"]))
        detections_by_frame = load_detections_from_raw(source)
        original_masks = [
            detection["mask"] for detections in detections_by_frame for detection in detections
        ]
        frames_total = len(detections_by_frame)
        image_shape = tuple(map(int, source["frame_image_shape_hw"][0]))
        timestamps = source["timestamp_sec"].astype(np.float64)
        frame_indices = source["frame_index"].astype(np.int32)
        class_names = {
            int(key): value for key, value in metadata["classes"].items()
        }

        original_tracks = tracks_from_detections(detections_by_frame)
        original_bench_details = [
            track_detail(
                track,
                next((summary.get("score_components", {}) for summary in original_summaries if int(summary["track_id"]) == track["track_id"]), {}),
                frames_total,
            )
            for track in original_tracks.values() if track["class_name"] == "bench"
        ]
        original_track_details = {
            class_name: [
                track_detail(
                    track,
                    next((
                        summary.get("score_components", {})
                        for summary in original_summaries
                        if int(summary["track_id"]) == track["track_id"]
                    ), {}),
                    frames_total,
                )
                for track in original_tracks.values()
                if track["class_name"] == class_name
            ]
            for class_name in class_names.values()
        }

        stitch_mapping, accepted_stitches, rejected_stitches = stitch_non_overlapping_bench_tracks(
            original_tracks, image_shape
        )
        for detections in detections_by_frame:
            for detection in detections:
                if detection["class_name"] == "bench":
                    detection["track_id"] = stitch_mapping.get(
                        detection["track_id"], detection["track_id"]
                    )
        stitched_tracks = tracks_from_detections(detections_by_frame)
        original_main_bench = int(original_selected["bench"])
        bench_anchor_id, bench_anchor_scores = select_bench_anchor(
            stitched_tracks, frames_total, image_shape
        )
        selected_bench_by_frame, identity_audit = consolidate_bench_identity(
            detections_by_frame, stitched_tracks, bench_anchor_id, image_shape
        )

        fallback_frames = identity_audit["fallback_frames"]
        virtual_main_track_id = bench_anchor_id
        if fallback_frames:
            virtual_main_track_id = max(stitched_tracks) + 1
            selected_indices = {
                detection["global_index"] for detection in selected_bench_by_frame.values()
            }
            for detections in detections_by_frame:
                for detection in detections:
                    if detection["global_index"] in selected_indices:
                        detection["track_id"] = virtual_main_track_id

        tracks_after_bench = tracks_from_detections(detections_by_frame)
        original_main_bar = int(original_selected["bar"])
        selected_bar_by_frame, bar_identity_audit = consolidate_bar_handoff(
            tracks_after_bench, original_main_bar, image_shape
        )
        virtual_main_bar_id = original_main_bar
        if bar_identity_audit["accepted_handoffs"]:
            virtual_main_bar_id = max(tracks_after_bench) + 1
            selected_bar_indices = {
                detection["global_index"]
                for detection in selected_bar_by_frame.values()
            }
            for detections in detections_by_frame:
                for detection in detections:
                    if detection["global_index"] in selected_bar_indices:
                        detection["track_id"] = virtual_main_bar_id

        tracks = tracks_from_detections(detections_by_frame)
        selected_tracks, selection_scores = select_main_tracks(
            tracks, class_names, frames_total, image_shape
        )
        selected_tracks["bench"] = virtual_main_track_id
        selected_tracks["bar"] = virtual_main_bar_id
        track_summaries = summarize_tracks(
            tracks, selection_scores, selected_tracks
        )
        for summary in track_summaries:
            track = tracks[int(summary["track_id"])]
            summary["original_track_ids"] = sorted({
                int(detection["original_track_id"]) for detection in track["detections"]
            })
            summary["identity_role"] = (
                "main_bench_identity" if summary["class_name"] == "bench"
                and summary["track_id"] == virtual_main_track_id
                else "residual_or_independent"
            )

        object_features, selected_detection_indices = build_object_features(
            detections_by_frame,
            tracks,
            selected_tracks,
            class_names,
            timestamps,
            image_shape,
        )
        detections_by_index = {
            detection["global_index"]: detection
            for detections in detections_by_frame for detection in detections
        }
        recalculate_temporal_features(
            object_features,
            selected_detection_indices,
            detections_by_index,
            timestamps,
            image_shape,
        )
        interaction_features = build_interaction_features(
            detections_by_frame,
            selected_detection_indices,
            class_names,
            float(metadata["proximity_threshold_normalized"]),
        )
        interaction_summary = summarize_interactions(interaction_features, timestamps)

        derivation_id = str(uuid4())
        derived_metadata = dict(metadata)
        derived_metadata["pipeline_version"] = PIPELINE_VERSION
        derived_metadata["tracking_original"] = metadata["tracking"]
        derived_metadata["tracking"] = {
            "algorithm_version": ALGORITHM_VERSION,
            "bar_person_policy": "original association plus strict audited bar handoff; person unchanged",
            "bench_stitch_max_gap_frames": 60,
            "bench_stitch_min_mask_iou": 0.80,
            "bench_stitch_min_bbox_iou": 0.80,
            "bench_stitch_max_centroid_distance_normalized": 0.01,
            "bench_stitch_max_area_ratio": 1.25,
            "bench_identity_min_mask_iou": 0.55,
            "bench_identity_min_containment": 0.95,
            "bench_identity_max_centroid_distance_normalized": 0.04,
            "bench_identity_max_area_ratio": 1.80,
            "bench_identity_min_shared_frames": BENCH_IDENTITY_MIN_SHARED_FRAMES,
            "bench_identity_min_joint_support_ratio": 0.75,
            "identity_fallback_interpolation": False,
            "bar_handoff_min_shared_frames": 2,
            "bar_handoff_max_shared_frames": 5,
            "bar_handoff_min_mask_iou_each_frame": 0.90,
            "bar_handoff_min_bbox_iou_each_frame": 0.65,
            "bar_handoff_min_containment_each_frame": 0.99,
            "bar_handoff_max_centroid_distance_normalized_each_frame": 0.01,
            "bar_handoff_max_area_ratio_each_frame": 1.10,
        }
        derived_metadata["derivation"] = {
            "derivation_id": derivation_id,
            "source_raw_path": str(source_raw_path.resolve()),
            "source_raw_sha256": source_hash_before,
            "source_raw_schema_version": str(source["schema_version"]),
            "source_extraction_id": metadata["extraction_id"],
            "script": Path(__file__).name,
            "algorithm_version": ALGORITHM_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "timestamp_utc": utc_now(),
            "uses_yolo": False,
        }
        derived_metadata["selected_tracks"] = selected_tracks

        merge_audit = {
            "algorithm_version": ALGORITHM_VERSION,
            "thresholds": derived_metadata["tracking"],
            "accepted_stitches": accepted_stitches,
            "rejected_stitches": rejected_stitches,
            "original_main_track_id": original_main_bench,
            "anchor_selection": {
                "selected_track_id": bench_anchor_id,
                "policy": "bench-static score: coverage 0.50, continuity 0.15, confidence 0.05, area 0.05, centrality 0.10, stability 0.15",
                "scores": bench_anchor_scores,
            },
            "bench_identity": identity_audit,
            "bar_identity": bar_identity_audit,
            "merged_track_id": int(virtual_main_track_id),
            "original_track_ids": sorted({
                int(detection["original_track_id"])
                for detection in tracks[virtual_main_track_id]["detections"]
            }),
            "criterion": "strict non-overlap stitching plus audited same-bench identity stitching",
        }

        derived_raw_arrays = dict(source_arrays)
        derived_raw_arrays.update({
            "schema_version": np.asarray(RETRACKED_RAW_SCHEMA_VERSION),
            "metadata_json": np.asarray(json_dumps(derived_metadata)),
            "track_summaries_json": np.asarray(json_dumps(track_summaries)),
            "selected_tracks_json": np.asarray(json_dumps(selected_tracks)),
            "instance_original_track_id": source["instance_track_id"].astype(np.int32).copy(),
            "instance_track_id": np.asarray([
                detections_by_index[index]["track_id"]
                for index in range(len(detections_by_index))
            ], dtype=np.int32),
            "original_track_summaries_json": np.asarray(str(source["track_summaries_json"])),
            "original_selected_tracks_json": np.asarray(str(source["selected_tracks_json"])),
            "track_merge_audit_json": np.asarray(json_dumps(merge_audit)),
            "source_raw_sha256": np.asarray(source_hash_before),
        })
        derived_feature_arrays = build_feature_arrays(
            frame_indices,
            timestamps,
            class_names,
            selected_tracks,
            object_features,
            selected_detection_indices,
            interaction_features,
            interaction_summary,
            derived_metadata,
        )
        derived_feature_arrays["schema_version"] = np.asarray(
            RETRACKED_FEATURES_SCHEMA_VERSION
        )

        before_names = old_features["interaction_feature_names"].astype(str).tolist()
        before_pairs = old_features["interaction_pair_names"].astype(str).tolist()
        before_both = old_features["interaction_features"][
            :, before_pairs.index("person_bench"), before_names.index("both_detected")
        ]
        after_both = interaction_features[
            :, ["_".join(pair) for pair in [("bar", "person"), ("person", "bench"), ("bar", "bench")]].index("person_bench"),
            INTERACTION_FEATURE_NAMES.index("both_detected"),
        ]

    output_raw_path.parent.mkdir(parents=True, exist_ok=True)
    output_features_path.parent.mkdir(parents=True, exist_ok=True)
    output_audit_path.parent.mkdir(parents=True, exist_ok=True)
    raw_temp = output_raw_path.with_suffix(output_raw_path.suffix + ".tmp")
    features_temp = output_features_path.with_suffix(output_features_path.suffix + ".tmp")
    raw_temp.unlink(missing_ok=True)
    features_temp.unlink(missing_ok=True)
    try:
        save_npz(raw_temp, derived_raw_arrays)
        save_npz(features_temp, derived_feature_arrays)
        validate_raw_file(raw_temp, original_masks)
        validate_features_file(features_temp, raw_temp)
        with np.load(source_raw_path, allow_pickle=False) as source, np.load(raw_temp, allow_pickle=False) as derived:
            preserved_arrays = source_preserved(source, derived)
        atomic_publish_pair(raw_temp, output_raw_path, features_temp, output_features_path)
    except Exception:
        raw_temp.unlink(missing_ok=True)
        features_temp.unlink(missing_ok=True)
        raise

    if sha256_file(source_raw_path) != source_hash_before:
        raise ValueError("El RAW fuente cambió durante la derivación")
    derived_validation = validate_raw_file(output_raw_path)
    validate_features_file(output_features_path, output_raw_path)

    final_tracks = tracks_from_detections(detections_by_frame)
    final_bench_details = [
        track_detail(
            track,
            next((summary.get("score_components", {}) for summary in track_summaries if int(summary["track_id"]) == track["track_id"]), {}),
            frames_total,
        )
        for track in final_tracks.values() if track["class_name"] == "bench"
    ]
    final_track_details = {
        class_name: [
            track_detail(
                track,
                next((
                    summary.get("score_components", {})
                    for summary in track_summaries
                    if int(summary["track_id"]) == track["track_id"]
                ), {}),
                frames_total,
            )
            for track in final_tracks.values()
            if track["class_name"] == class_name
        ]
        for class_name in class_names.values()
    }
    audit = {
        "video_id": video_id,
        "source_raw_path": str(source_raw_path.resolve()),
        "source_raw_sha256": source_hash_before,
        "source_raw_unchanged": True,
        "uses_yolo": False,
        "roundtrip_masks": int(derived_validation["roundtrip_masks"]),
        "immutable_arrays_verified": preserved_arrays,
        "original_selected_tracks": original_selected,
        "selected_tracks_after": selected_tracks,
        "bench_tracks_before": original_bench_details,
        "bench_tracks_after": final_bench_details,
        "tracks_before_by_class": original_track_details,
        "tracks_after_by_class": final_track_details,
        "merge_audit": merge_audit,
    }
    atomic_write_json(output_audit_path, audit)

    original_by_class = {
        name: [summary for summary in original_summaries if summary["class_name"] == name]
        for name in class_names.values()
    }
    final_by_class = {
        name: [summary for summary in track_summaries if summary["class_name"] == name]
        for name in class_names.values()
    }
    comparison_rows = []
    for class_name in class_names.values():
        old_main = int(original_selected[class_name])
        new_main = int(selected_tracks[class_name])
        old_summary = next(summary for summary in original_by_class[class_name] if int(summary["track_id"]) == old_main)
        new_summary = next(summary for summary in final_by_class[class_name] if int(summary["track_id"]) == new_main)
        old_coverage = int(old_summary["frames_covered"])
        new_coverage = int(new_summary["frames_covered"])
        comparison_rows.append({
            "video_id": video_id,
            "class_name": class_name,
            "frames_total": frames_total,
            "raw_instances": len(detections_by_index),
            "tracks_before": len(original_by_class[class_name]),
            "tracks_after": len(final_by_class[class_name]),
            "main_track_before": old_main,
            "main_track_after": new_main,
            "main_coverage_before": old_coverage / frames_total,
            "main_coverage_after": new_coverage / frames_total,
            "main_missing_before": frames_total - old_coverage,
            "main_missing_after": frames_total - new_coverage,
            "person_bench_frames_before": int(np.count_nonzero(before_both == 1.0)),
            "person_bench_frames_after": int(np.count_nonzero(after_both == 1.0)),
            "accepted_stitches": len(accepted_stitches),
            "identity_fallback_frames": len(fallback_frames),
            "accepted_bar_handoffs": len(bar_identity_audit["accepted_handoffs"]),
            "source_raw_size_bytes": source_raw_path.stat().st_size,
            "derived_raw_size_bytes": output_raw_path.stat().st_size,
            "features_before_size_bytes": source_features_path.stat().st_size,
            "features_after_size_bytes": output_features_path.stat().st_size,
            "source_raw_sha256": source_hash_before,
            "source_raw_unchanged": 1,
            "roundtrip_masks": derived_validation["roundtrip_masks"],
        })

    print(
        "[OK] %s: bench %.3f -> %.3f, fallback=%d, YOLO=False"
        % (
            video_id,
            next(row["main_coverage_before"] for row in comparison_rows if row["class_name"] == "bench"),
            next(row["main_coverage_after"] for row in comparison_rows if row["class_name"] == "bench"),
            len(fallback_frames),
        )
    )
    return comparison_rows


def normalize_video_id(value):
    stem = Path(value).stem
    if stem.isdigit():
        return "bp_%03d" % int(stem)
    if not stem.startswith("bp_"):
        raise ValueError("ID no reconocido: %s" % value)
    return stem


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenera tracking y features BP_FRONT desde RAW, sin YOLO"
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--videos", nargs="+")
    selection.add_argument(
        "--all", action="store_true",
        help="Procesa todos los RAW fuente bp_### disponibles",
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.source_dir.resolve() == args.output_dir.resolve():
        raise ValueError("source-dir y output-dir deben ser distintos")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / "manifests" / "bp_front_tracking_v2_comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    values = args.videos
    if args.all:
        values = sorted(
            path.name.removesuffix("_segmentation.npz")
            for path in (args.source_dir / "raw").glob("bp_*_segmentation.npz")
        )
    for value in values:
        video_id = normalize_video_id(value)
        try:
            rows = process_video(
                video_id, args.source_dir, args.output_dir, args.overwrite
            )
            if rows is not None:
                upsert_comparison(comparison_path, rows)
        except Exception as error:
            failures += 1
            print("[ERROR] %s: %s" % (video_id, error))
    print("Reprocesamiento RAW -> tracking/features terminado; fallidos=%d" % failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
