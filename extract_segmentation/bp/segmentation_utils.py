"""Utilidades compartidas de segmentación BP front y formatos RAW/features.

Codifica y valida máscaras, asocia detecciones entre frames, elige tracks
principales y construye geometría e interacciones. Es un módulo de apoyo:
no procesa vídeos por sí solo ni entrena modelos.
"""

import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


RAW_SCHEMA_VERSION = "bp_front_segmentation_raw_v2"
FEATURES_SCHEMA_VERSION = "bp_front_segmentation_features_v2"
RETRACKED_RAW_SCHEMA_VERSION = "bp_front_segmentation_raw_v3_retracked"
RETRACKED_FEATURES_SCHEMA_VERSION = "bp_front_segmentation_features_v3_retracked"

OBJECT_FEATURE_NAMES = [
    "detected",
    "raw_instance_index",
    "track_id",
    "confidence",
    "bbox_x1_px",
    "bbox_y1_px",
    "bbox_x2_px",
    "bbox_y2_px",
    "bbox_cx_px",
    "bbox_cy_px",
    "bbox_width_px",
    "bbox_height_px",
    "bbox_x1_normalized",
    "bbox_y1_normalized",
    "bbox_x2_normalized",
    "bbox_y2_normalized",
    "bbox_cx_normalized",
    "bbox_cy_normalized",
    "bbox_width_normalized",
    "bbox_height_normalized",
    "mask_area_px",
    "mask_area_normalized",
    "mask_centroid_x_px",
    "mask_centroid_y_px",
    "mask_centroid_x_normalized",
    "mask_centroid_y_normalized",
    "topmost_y_px",
    "bottommost_y_px",
    "leftmost_x_px",
    "rightmost_x_px",
    "topmost_y_normalized",
    "bottommost_y_normalized",
    "leftmost_x_normalized",
    "rightmost_x_normalized",
    "orientation_angle_deg",
    "major_axis_length_px",
    "minor_axis_length_px",
    "major_axis_length_normalized",
    "minor_axis_length_normalized",
    "dt_seconds",
    "centroid_dx_px",
    "centroid_dy_px",
    "centroid_dx_normalized",
    "centroid_dy_normalized",
    "centroid_vx_px_s",
    "centroid_vy_px_s",
    "centroid_vx_normalized_s",
    "centroid_vy_normalized_s",
    "centroid_ax_px_s2",
    "centroid_ay_px_s2",
    "centroid_ax_normalized_s2",
    "centroid_ay_normalized_s2",
]

INTERACTION_FEATURE_NAMES = [
    "both_detected",
    "centroid_distance_px",
    "centroid_distance_normalized",
    "relative_dx_px",
    "relative_dy_px",
    "relative_dx_normalized",
    "relative_dy_normalized",
    "minimum_mask_distance_px",
    "minimum_mask_distance_normalized",
    "intersection_area_px",
    "intersection_area_normalized",
    "union_area_px",
    "union_area_normalized",
    "iou",
    "a_contained_in_b_ratio",
    "b_contained_in_a_ratio",
    "overlap_detected",
    "proximity_candidate",
]

INTERACTION_PAIRS = [
    ("bar", "person"),
    ("person", "bench"),
    ("bar", "bench"),
]


def json_dumps(data):
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def longest_missing_gap(detected):
    longest = 0
    current = 0
    for value in np.asarray(detected, dtype=bool):
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return int(longest)


def finite_stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def mask_geometry(mask):
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    ys, xs = np.nonzero(mask_u8)
    if xs.size == 0:
        return {
            "area": 0,
            "centroid_x": np.nan,
            "centroid_y": np.nan,
            "topmost_y": np.nan,
            "bottommost_y": np.nan,
            "leftmost_x": np.nan,
            "rightmost_x": np.nan,
            "orientation_angle_deg": np.nan,
            "major_axis_length": np.nan,
            "minor_axis_length": np.nan,
        }

    moments = cv2.moments(mask_u8, binaryImage=True)
    area = int(xs.size)
    centroid_x = float(moments["m10"] / moments["m00"])
    centroid_y = float(moments["m01"] / moments["m00"])

    if area >= 2:
        covariance = np.cov(np.column_stack((xs, ys)), rowvar=False, bias=True)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        vector = eigenvectors[:, order[0]]
        angle = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
        while angle >= 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0
        major_axis = 4.0 * math.sqrt(float(eigenvalues[0]))
        minor_axis = 4.0 * math.sqrt(float(eigenvalues[1]))
    else:
        angle = np.nan
        major_axis = np.nan
        minor_axis = np.nan

    return {
        "area": area,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "topmost_y": float(ys.min()),
        "bottommost_y": float(ys.max()),
        "leftmost_x": float(xs.min()),
        "rightmost_x": float(xs.max()),
        "orientation_angle_deg": float(angle),
        "major_axis_length": float(major_axis),
        "minor_axis_length": float(minor_axis),
    }


def pack_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return (0, 0), (0, 0), np.empty(0, dtype=np.uint8)
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    crop = np.ascontiguousarray(mask[y0:y1, x0:x1])
    packed = np.packbits(crop.reshape(-1), bitorder="little")
    return (x0, y0), crop.shape, packed


def unpack_mask(packed, origin_xy, shape_hw, image_shape_hw):
    image_height, image_width = map(int, image_shape_hw)
    x0, y0 = map(int, origin_xy)
    crop_height, crop_width = map(int, shape_hw)
    result = np.zeros((image_height, image_width), dtype=bool)
    if crop_height == 0 or crop_width == 0:
        return result
    count = crop_height * crop_width
    crop = np.unpackbits(
        np.asarray(packed, dtype=np.uint8), bitorder="little", count=count
    ).reshape(crop_height, crop_width).astype(bool)
    result[y0:y0 + crop_height, x0:x0 + crop_width] = crop
    return result


def reconstruct_mask(raw_data, instance_index):
    start = int(raw_data["mask_byte_offsets"][instance_index])
    end = int(raw_data["mask_byte_offsets"][instance_index + 1])
    frame_index = int(raw_data["instance_frame_index"][instance_index])
    image_shape = raw_data["frame_image_shape_hw"][frame_index]
    return unpack_mask(
        raw_data["mask_bits"][start:end],
        raw_data["instance_mask_origin_xy"][instance_index],
        raw_data["instance_mask_shape_hw"][instance_index],
        image_shape,
    )


def load_detections_from_raw(raw_data):
    frames_total = len(raw_data["frame_index"])
    detections_by_frame = [[] for _ in range(frames_total)]
    polygon_offsets = raw_data["polygon_point_offsets"].astype(np.int64)

    for instance_index in range(len(raw_data["instance_frame_index"])):
        frame_index = int(raw_data["instance_frame_index"][instance_index])
        mask = reconstruct_mask(raw_data, instance_index)
        geometry = mask_geometry(mask)
        if geometry["area"] != int(raw_data["instance_mask_area_px"][instance_index]):
            raise ValueError("Área RAW inconsistente en instancia %d" % instance_index)
        polygon_start = int(polygon_offsets[instance_index])
        polygon_end = int(polygon_offsets[instance_index + 1])
        original_track_id = int(raw_data["instance_track_id"][instance_index])
        detections_by_frame[frame_index].append({
            "global_index": instance_index,
            "frame_index": frame_index,
            "instance_index": int(raw_data["instance_index_in_frame"][instance_index]),
            "class_id": int(raw_data["instance_class_id"][instance_index]),
            "class_name": str(raw_data["instance_class_name"][instance_index]),
            "confidence": float(raw_data["instance_confidence"][instance_index]),
            "bbox_xyxy": raw_data["instance_bbox_xyxy_px"][instance_index].astype(np.float32).copy(),
            "mask": mask,
            "polygon_xy": raw_data["polygon_xy_px"][polygon_start:polygon_end].astype(np.float32).copy(),
            "geometry": geometry,
            "original_track_id": original_track_id,
            "track_id": original_track_id,
        })
    return detections_by_frame


def bbox_iou(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def mask_iou(mask_a, mask_b):
    intersection = int(np.count_nonzero(mask_a & mask_b))
    union = int(np.count_nonzero(mask_a | mask_b))
    return intersection / union if union else 0.0


def _tracking_cost(previous, current, diagonal):
    previous_center = np.array([previous["geometry"]["centroid_x"], previous["geometry"]["centroid_y"]])
    current_center = np.array([current["geometry"]["centroid_x"], current["geometry"]["centroid_y"]])
    center_distance = float(np.linalg.norm(current_center - previous_center) / diagonal)
    area_ratio = max(previous["geometry"]["area"], 1) / max(current["geometry"]["area"], 1)
    area_change = abs(math.log(area_ratio))
    box_overlap = bbox_iou(previous["bbox_xyxy"], current["bbox_xyxy"])
    mask_overlap = mask_iou(previous["mask"], current["mask"])

    allowed = (
        center_distance <= 0.15
        and area_change <= math.log(6.0)
        and (mask_overlap > 0.0 or box_overlap > 0.0 or center_distance <= 0.04)
    )
    if not allowed:
        return np.inf

    return (
        0.45 * (1.0 - mask_overlap)
        + 0.20 * (1.0 - box_overlap)
        + 0.25 * min(center_distance / 0.15, 1.0)
        + 0.10 * min(area_change / math.log(6.0), 1.0)
    )


def track_detections(
    detections_by_frame,
    class_names,
    image_shape_hw,
    max_gap=2,
    max_cost=0.85,
    stationary_max_gap=60,
):
    image_height, image_width = map(int, image_shape_hw)
    diagonal = math.hypot(image_width, image_height)
    tracks = {}
    next_track_id = 0

    for frame_index, detections in enumerate(detections_by_frame):
        for class_id in sorted(class_names):
            current = [detection for detection in detections if detection["class_id"] == class_id]
            class_max_gap = (
                max(max_gap, stationary_max_gap)
                if class_names[class_id] == "bench"
                else max_gap
            )
            active = [
                track for track in tracks.values()
                if track["class_id"] == class_id
                and frame_index - track["last_frame"] <= class_max_gap + 1
            ]
            matched_current = set()

            if active and current:
                costs = np.full((len(active), len(current)), np.inf, dtype=np.float64)
                for row, track in enumerate(active):
                    previous = track["detections"][-1]
                    gap = frame_index - track["last_frame"] - 1
                    for column, detection in enumerate(current):
                        cost = _tracking_cost(previous, detection, diagonal)
                        if np.isfinite(cost):
                            gap_penalty = 0.005 if class_names[class_id] == "bench" else 0.05
                            costs[row, column] = cost + min(gap_penalty * gap, 0.20)

                allowed_costs = np.isfinite(costs) & (costs <= max_cost)
                assignment_costs = np.where(allowed_costs, costs, 1e6)
                dummy_cost = max_cost + 1e-6
                assignment_costs = np.column_stack((
                    assignment_costs,
                    np.full((len(active), len(active)), dummy_cost),
                ))
                rows, columns = linear_sum_assignment(assignment_costs)
                for row, column in zip(rows, columns):
                    if column >= len(current) or not allowed_costs[row, column]:
                        continue
                    track = active[row]
                    detection = current[column]
                    detection["track_id"] = track["track_id"]
                    track["detections"].append(detection)
                    track["last_frame"] = frame_index
                    matched_current.add(column)

            for column, detection in enumerate(current):
                if column in matched_current:
                    continue
                detection["track_id"] = next_track_id
                tracks[next_track_id] = {
                    "track_id": next_track_id,
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "first_frame": frame_index,
                    "last_frame": frame_index,
                    "detections": [detection],
                }
                next_track_id += 1

    return tracks


def _track_components(track, frames_total, image_shape_hw, related_tracks=None):
    image_height, image_width = map(int, image_shape_hw)
    diagonal = math.hypot(image_width, image_height)
    detections = track["detections"]
    frames = np.array([detection["frame_index"] for detection in detections], dtype=np.int32)
    confidences = np.array([detection["confidence"] for detection in detections], dtype=np.float64)
    areas = np.array([detection["geometry"]["area"] for detection in detections], dtype=np.float64)
    centers = np.array([
        [detection["geometry"]["centroid_x"], detection["geometry"]["centroid_y"]]
        for detection in detections
    ])
    coverage = len(np.unique(frames)) / max(frames_total, 1)
    span = int(frames.max() - frames.min() + 1)
    continuity = len(np.unique(frames)) / max(span, 1)
    confidence = float(np.mean(confidences))
    center = np.array([image_width / 2.0, image_height / 2.0])
    centrality = 1.0 - min(float(np.median(np.linalg.norm(centers - center, axis=1))) / (0.5 * diagonal), 1.0)
    if len(centers) > 1:
        stability = 1.0 - min(float(np.median(np.linalg.norm(np.diff(centers, axis=0), axis=1))) / (0.10 * diagonal), 1.0)
    else:
        stability = 0.0

    relation = 0.0
    if related_tracks:
        related_by_frame = {}
        for related_track in related_tracks:
            for detection in related_track["detections"]:
                related_by_frame.setdefault(detection["frame_index"], []).append(detection)
        closeness = []
        for detection, detection_center in zip(detections, centers):
            candidates = related_by_frame.get(detection["frame_index"], [])
            if candidates:
                distances = [
                    np.linalg.norm(detection_center - np.array([
                        candidate["geometry"]["centroid_x"],
                        candidate["geometry"]["centroid_y"],
                    ])) / diagonal
                    for candidate in candidates
                ]
                closeness.append(1.0 - min(min(distances) / 0.5, 1.0))
        if closeness:
            relation = float(np.mean(closeness)) * (len(closeness) / len(detections))

    angles = np.array([
        abs(detection["geometry"]["orientation_angle_deg"])
        for detection in detections
        if np.isfinite(detection["geometry"]["orientation_angle_deg"])
    ])
    horizontalness = 1.0 - min(float(np.median(angles)) / 90.0, 1.0) if angles.size else 0.0

    return {
        "coverage": float(coverage),
        "continuity": float(continuity),
        "mean_confidence": confidence,
        "median_area_normalized": float(np.median(areas) / (image_width * image_height)),
        "centrality": float(centrality),
        "stability": float(stability),
        "relationship": float(relation),
        "horizontalness": float(horizontalness),
    }


def _select_track(candidates, frames_total, image_shape_hw, class_name, related_tracks=None):
    if not candidates:
        return -1, {}
    components_by_track = {}
    max_area = 0.0
    for track in candidates:
        components = _track_components(track, frames_total, image_shape_hw, related_tracks)
        components_by_track[track["track_id"]] = components
        max_area = max(max_area, components["median_area_normalized"])

    for track_id, components in components_by_track.items():
        components["relative_area"] = components["median_area_normalized"] / max_area if max_area > 0 else 0.0
        if class_name == "bench":
            score = (
                0.38 * components["coverage"]
                + 0.10 * components["continuity"]
                + 0.20 * components["mean_confidence"]
                + 0.02 * components["relative_area"]
                + 0.15 * components["centrality"]
                + 0.15 * components["stability"]
            )
        elif class_name == "bar":
            score = (
                0.35 * components["coverage"]
                + 0.15 * components["continuity"]
                + 0.10 * components["mean_confidence"]
                + 0.05 * components["relative_area"]
                + 0.10 * components["centrality"]
                + 0.05 * components["stability"]
                + 0.10 * components["relationship"]
                + 0.10 * components["horizontalness"]
            )
        else:
            score = (
                0.30 * components["coverage"]
                + 0.10 * components["continuity"]
                + 0.05 * components["mean_confidence"]
                + 0.10 * components["relative_area"]
                + 0.10 * components["centrality"]
                + 0.05 * components["stability"]
                + 0.30 * components["relationship"]
            )
        components["selection_score"] = float(score)

    selected = max(
        candidates,
        key=lambda track: (
            components_by_track[track["track_id"]]["selection_score"],
            components_by_track[track["track_id"]]["coverage"],
            -track["track_id"],
        ),
    )
    return selected["track_id"], components_by_track


def select_main_tracks(tracks, class_names, frames_total, image_shape_hw):
    by_name = {
        name: [track for track in tracks.values() if track["class_name"] == name]
        for name in class_names.values()
    }
    selected = {}
    scores = {}

    bench_id, bench_scores = _select_track(by_name.get("bench", []), frames_total, image_shape_hw, "bench")
    selected["bench"] = bench_id
    scores["bench"] = bench_scores
    bench_related = [tracks[bench_id]] if bench_id >= 0 else []

    bar_id, bar_scores = _select_track(by_name.get("bar", []), frames_total, image_shape_hw, "bar", bench_related)
    selected["bar"] = bar_id
    scores["bar"] = bar_scores
    person_related = list(bench_related)
    if bar_id >= 0:
        person_related.append(tracks[bar_id])

    person_id, person_scores = _select_track(by_name.get("person", []), frames_total, image_shape_hw, "person", person_related)
    selected["person"] = person_id
    scores["person"] = person_scores
    return selected, scores


def summarize_tracks(tracks, selection_scores, selected_tracks):
    summaries = []
    for track_id in sorted(tracks):
        track = tracks[track_id]
        detections = track["detections"]
        components = selection_scores.get(track["class_name"], {}).get(track_id, {})
        summaries.append({
            "track_id": int(track_id),
            "class_id": int(track["class_id"]),
            "class_name": track["class_name"],
            "first_frame": int(track["first_frame"]),
            "last_frame": int(track["last_frame"]),
            "duration_frames": int(track["last_frame"] - track["first_frame"] + 1),
            "frames_covered": int(len(detections)),
            "mean_confidence": float(np.mean([detection["confidence"] for detection in detections])),
            "selected_as_main": bool(selected_tracks.get(track["class_name"], -1) == track_id),
            "score_components": components,
        })
    return summaries


def _feature_index(name):
    return OBJECT_FEATURE_NAMES.index(name)


def build_object_features(detections_by_frame, tracks, selected_tracks, class_names, timestamps, image_shape_hw):
    frames_total = len(detections_by_frame)
    class_ids = sorted(class_names)
    image_height, image_width = map(int, image_shape_hw)
    diagonal = math.hypot(image_width, image_height)
    image_area = image_width * image_height
    features = np.full((frames_total, len(class_ids), len(OBJECT_FEATURE_NAMES)), np.nan, dtype=np.float64)
    features[:, :, _feature_index("detected")] = 0.0
    selected_detection_indices = np.full((frames_total, len(class_ids)), -1, dtype=np.int64)

    class_position = {class_id: position for position, class_id in enumerate(class_ids)}
    selected_ids = {class_id: selected_tracks.get(class_names[class_id], -1) for class_id in class_ids}

    for frame_index, detections in enumerate(detections_by_frame):
        for detection in detections:
            class_id = detection["class_id"]
            if detection["track_id"] != selected_ids[class_id]:
                continue
            position = class_position[class_id]
            geometry = detection["geometry"]
            x1, y1, x2, y2 = map(float, detection["bbox_xyxy"])
            width = x2 - x1
            height = y2 - y1
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            values = {
                "detected": 1.0,
                "raw_instance_index": float(detection["global_index"]),
                "track_id": float(detection["track_id"]),
                "confidence": float(detection["confidence"]),
                "bbox_x1_px": x1,
                "bbox_y1_px": y1,
                "bbox_x2_px": x2,
                "bbox_y2_px": y2,
                "bbox_cx_px": cx,
                "bbox_cy_px": cy,
                "bbox_width_px": width,
                "bbox_height_px": height,
                "bbox_x1_normalized": x1 / image_width,
                "bbox_y1_normalized": y1 / image_height,
                "bbox_x2_normalized": x2 / image_width,
                "bbox_y2_normalized": y2 / image_height,
                "bbox_cx_normalized": cx / image_width,
                "bbox_cy_normalized": cy / image_height,
                "bbox_width_normalized": width / image_width,
                "bbox_height_normalized": height / image_height,
                "mask_area_px": float(geometry["area"]),
                "mask_area_normalized": geometry["area"] / image_area,
                "mask_centroid_x_px": geometry["centroid_x"],
                "mask_centroid_y_px": geometry["centroid_y"],
                "mask_centroid_x_normalized": geometry["centroid_x"] / image_width,
                "mask_centroid_y_normalized": geometry["centroid_y"] / image_height,
                "topmost_y_px": geometry["topmost_y"],
                "bottommost_y_px": geometry["bottommost_y"],
                "leftmost_x_px": geometry["leftmost_x"],
                "rightmost_x_px": geometry["rightmost_x"],
                "topmost_y_normalized": geometry["topmost_y"] / image_height,
                "bottommost_y_normalized": geometry["bottommost_y"] / image_height,
                "leftmost_x_normalized": geometry["leftmost_x"] / image_width,
                "rightmost_x_normalized": geometry["rightmost_x"] / image_width,
                "orientation_angle_deg": geometry["orientation_angle_deg"],
                "major_axis_length_px": geometry["major_axis_length"],
                "minor_axis_length_px": geometry["minor_axis_length"],
                "major_axis_length_normalized": geometry["major_axis_length"] / diagonal,
                "minor_axis_length_normalized": geometry["minor_axis_length"] / diagonal,
            }
            for name, value in values.items():
                features[frame_index, position, _feature_index(name)] = value
            selected_detection_indices[frame_index, position] = detection["global_index"]

    centroid_x = _feature_index("mask_centroid_x_px")
    centroid_y = _feature_index("mask_centroid_y_px")
    detected_index = _feature_index("detected")
    vx_index = _feature_index("centroid_vx_px_s")
    vy_index = _feature_index("centroid_vy_px_s")
    vx_norm_index = _feature_index("centroid_vx_normalized_s")
    vy_norm_index = _feature_index("centroid_vy_normalized_s")

    for class_position_index in range(len(class_ids)):
        for frame_index in range(1, frames_total):
            if features[frame_index, class_position_index, detected_index] != 1.0:
                continue
            if features[frame_index - 1, class_position_index, detected_index] != 1.0:
                continue
            dt = float(timestamps[frame_index] - timestamps[frame_index - 1])
            if not np.isfinite(dt) or dt <= 0:
                continue
            dx = features[frame_index, class_position_index, centroid_x] - features[frame_index - 1, class_position_index, centroid_x]
            dy = features[frame_index, class_position_index, centroid_y] - features[frame_index - 1, class_position_index, centroid_y]
            temporal_values = {
                "dt_seconds": dt,
                "centroid_dx_px": dx,
                "centroid_dy_px": dy,
                "centroid_dx_normalized": dx / image_width,
                "centroid_dy_normalized": dy / image_height,
                "centroid_vx_px_s": dx / dt,
                "centroid_vy_px_s": dy / dt,
                "centroid_vx_normalized_s": (dx / image_width) / dt,
                "centroid_vy_normalized_s": (dy / image_height) / dt,
            }
            for name, value in temporal_values.items():
                features[frame_index, class_position_index, _feature_index(name)] = value

            if frame_index < 2:
                continue
            previous_vx = features[frame_index - 1, class_position_index, vx_index]
            previous_vy = features[frame_index - 1, class_position_index, vy_index]
            previous_vx_norm = features[frame_index - 1, class_position_index, vx_norm_index]
            previous_vy_norm = features[frame_index - 1, class_position_index, vy_norm_index]
            if not np.all(np.isfinite([previous_vx, previous_vy, previous_vx_norm, previous_vy_norm])):
                continue
            accelerations = {
                "centroid_ax_px_s2": (temporal_values["centroid_vx_px_s"] - previous_vx) / dt,
                "centroid_ay_px_s2": (temporal_values["centroid_vy_px_s"] - previous_vy) / dt,
                "centroid_ax_normalized_s2": (temporal_values["centroid_vx_normalized_s"] - previous_vx_norm) / dt,
                "centroid_ay_normalized_s2": (temporal_values["centroid_vy_normalized_s"] - previous_vy_norm) / dt,
            }
            for name, value in accelerations.items():
                features[frame_index, class_position_index, _feature_index(name)] = value

    return features, selected_detection_indices


def minimum_mask_distance(mask_a, mask_b, return_points=False):
    overlap = mask_a & mask_b
    if np.any(overlap):
        ys, xs = np.nonzero(overlap)
        point = (int(xs[0]), int(ys[0]))
        return (0.0, point, point) if return_points else 0.0

    ys_a, xs_a = np.nonzero(mask_a)
    ys_b, xs_b = np.nonzero(mask_b)
    if xs_a.size == 0 or xs_b.size == 0:
        return (np.nan, None, None) if return_points else np.nan

    x0 = int(min(xs_a.min(), xs_b.min()))
    y0 = int(min(ys_a.min(), ys_b.min()))
    x1 = int(max(xs_a.max(), xs_b.max())) + 1
    y1 = int(max(ys_a.max(), ys_b.max())) + 1
    roi_a = mask_a[y0:y1, x0:x1]
    roi_b = mask_b[y0:y1, x0:x1]
    distance_map = cv2.distanceTransform(
        (~roi_a).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    masked_distances = np.where(roi_b, distance_map, np.inf)
    flat_index = int(np.argmin(masked_distances))
    distance = float(masked_distances.reshape(-1)[flat_index])
    if not return_points:
        return distance

    by, bx = np.unravel_index(flat_index, masked_distances.shape)
    point_b_x = int(bx + x0)
    point_b_y = int(by + y0)
    squared_distances = (
        (xs_a.astype(np.int64) - point_b_x) ** 2
        + (ys_a.astype(np.int64) - point_b_y) ** 2
    )
    nearest_a = int(np.argmin(squared_distances))
    point_a = (int(xs_a[nearest_a]), int(ys_a[nearest_a]))
    return distance, point_a, (point_b_x, point_b_y)


def compute_mask_interaction(mask_a, mask_b, centroid_a, centroid_b, proximity_threshold_normalized):
    image_height, image_width = mask_a.shape
    diagonal = math.hypot(image_width, image_height)
    image_area = image_width * image_height
    area_a = int(np.count_nonzero(mask_a))
    area_b = int(np.count_nonzero(mask_b))
    intersection = int(np.count_nonzero(mask_a & mask_b))
    union = area_a + area_b - intersection
    distance = 0.0 if intersection > 0 else minimum_mask_distance(mask_a, mask_b)
    relative_dx = float(centroid_b[0] - centroid_a[0])
    relative_dy = float(centroid_b[1] - centroid_a[1])
    centroid_distance = math.hypot(relative_dx, relative_dy)
    return {
        "both_detected": 1.0,
        "centroid_distance_px": centroid_distance,
        "centroid_distance_normalized": centroid_distance / diagonal,
        "relative_dx_px": relative_dx,
        "relative_dy_px": relative_dy,
        "relative_dx_normalized": relative_dx / image_width,
        "relative_dy_normalized": relative_dy / image_height,
        "minimum_mask_distance_px": distance,
        "minimum_mask_distance_normalized": distance / diagonal,
        "intersection_area_px": float(intersection),
        "intersection_area_normalized": intersection / image_area,
        "union_area_px": float(union),
        "union_area_normalized": union / image_area,
        "iou": intersection / union if union else 0.0,
        "a_contained_in_b_ratio": intersection / area_a if area_a else np.nan,
        "b_contained_in_a_ratio": intersection / area_b if area_b else np.nan,
        "overlap_detected": float(intersection > 0),
        "proximity_candidate": float(distance / diagonal <= proximity_threshold_normalized),
    }


def compute_optional_region_interaction(bar_mask, bar_centroid, torso_mask=None, torso_roi=None, proximity_threshold_normalized=0.01):
    if (torso_mask is None) == (torso_roi is None):
        if torso_mask is None:
            return None
        raise ValueError("Proporcione torso_mask o torso_roi, no ambos")
    bar_mask = np.asarray(bar_mask, dtype=bool)
    if bar_mask.ndim != 2 or not np.any(bar_mask):
        raise ValueError("bar_mask debe ser una máscara 2D no vacía")
    if not np.all(np.isfinite(bar_centroid)):
        raise ValueError("bar_centroid debe contener coordenadas finitas")
    if not np.isfinite(proximity_threshold_normalized) or proximity_threshold_normalized < 0:
        raise ValueError("proximity_threshold_normalized inválido")
    if torso_mask is None and torso_roi is None:
        return None
    if torso_mask is None:
        x1, y1, x2, y2 = map(int, torso_roi)
        height, width = bar_mask.shape
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("torso_roi fuera de los límites de la imagen")
        torso_mask = np.zeros_like(bar_mask, dtype=bool)
        torso_mask[y1:y2, x1:x2] = True
    else:
        torso_mask = np.asarray(torso_mask, dtype=bool)
        if torso_mask.shape != bar_mask.shape:
            raise ValueError("torso_mask y bar_mask deben tener el mismo shape")
    torso_geometry = mask_geometry(torso_mask)
    if torso_geometry["area"] == 0:
        return None
    return compute_mask_interaction(
        bar_mask,
        np.asarray(torso_mask, dtype=bool),
        bar_centroid,
        (torso_geometry["centroid_x"], torso_geometry["centroid_y"]),
        proximity_threshold_normalized,
    )


def build_interaction_features(detections_by_frame, selected_detection_indices, class_names, proximity_threshold_normalized):
    frames_total = len(detections_by_frame)
    class_ids = sorted(class_names)
    class_position = {class_names[class_id]: position for position, class_id in enumerate(class_ids)}
    by_global_index = {
        detection["global_index"]: detection
        for detections in detections_by_frame
        for detection in detections
    }
    features = np.full(
        (frames_total, len(INTERACTION_PAIRS), len(INTERACTION_FEATURE_NAMES)),
        np.nan,
        dtype=np.float64,
    )
    features[:, :, INTERACTION_FEATURE_NAMES.index("both_detected")] = 0.0
    features[:, :, INTERACTION_FEATURE_NAMES.index("overlap_detected")] = 0.0
    features[:, :, INTERACTION_FEATURE_NAMES.index("proximity_candidate")] = 0.0

    for frame_index in range(frames_total):
        for pair_index, (name_a, name_b) in enumerate(INTERACTION_PAIRS):
            if name_a not in class_position or name_b not in class_position:
                continue
            index_a = int(selected_detection_indices[frame_index, class_position[name_a]])
            index_b = int(selected_detection_indices[frame_index, class_position[name_b]])
            if index_a < 0 or index_b < 0:
                continue
            detection_a = by_global_index[index_a]
            detection_b = by_global_index[index_b]
            geometry_a = detection_a["geometry"]
            geometry_b = detection_b["geometry"]
            values = compute_mask_interaction(
                detection_a["mask"],
                detection_b["mask"],
                (geometry_a["centroid_x"], geometry_a["centroid_y"]),
                (geometry_b["centroid_x"], geometry_b["centroid_y"]),
                proximity_threshold_normalized,
            )
            for name, value in values.items():
                features[frame_index, pair_index, INTERACTION_FEATURE_NAMES.index(name)] = value
    return features


def summarize_interactions(interaction_features, timestamps):
    summary = {}
    distance_index = INTERACTION_FEATURE_NAMES.index("minimum_mask_distance_px")
    iou_index = INTERACTION_FEATURE_NAMES.index("iou")
    overlap_index = INTERACTION_FEATURE_NAMES.index("overlap_detected")
    proximity_index = INTERACTION_FEATURE_NAMES.index("proximity_candidate")
    both_index = INTERACTION_FEATURE_NAMES.index("both_detected")

    for pair_index, pair in enumerate(INTERACTION_PAIRS):
        pair_name = "_".join(pair)
        values = interaction_features[:, pair_index]
        both = values[:, both_index] == 1.0
        proximity = both & (values[:, proximity_index] == 1.0)
        overlap = both & (values[:, overlap_index] == 1.0)
        proximity_indices = np.flatnonzero(proximity)
        longest_run = 0
        current_run = 0
        episodes = 0
        previous = False
        for value in proximity:
            if value:
                current_run += 1
                longest_run = max(longest_run, current_run)
                if not previous:
                    episodes += 1
            else:
                current_run = 0
            previous = bool(value)
        dt = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.0
        summary[pair_name] = {
            "frames_both_detected": int(np.count_nonzero(both)),
            "distance_px": finite_stats(values[both, distance_index]),
            "iou": finite_stats(values[both, iou_index]),
            "overlap_frames": int(np.count_nonzero(overlap)),
            "proximity_frames": int(np.count_nonzero(proximity)),
            "proximity_ratio_when_detected": float(np.mean(proximity[both])) if np.any(both) else None,
            "proximity_first_frame": int(proximity_indices[0]) if proximity_indices.size else None,
            "proximity_last_frame": int(proximity_indices[-1]) if proximity_indices.size else None,
            "proximity_first_time": float(timestamps[proximity_indices[0]]) if proximity_indices.size else None,
            "proximity_last_time": float(timestamps[proximity_indices[-1]]) if proximity_indices.size else None,
            "proximity_total_seconds": float(np.count_nonzero(proximity) * dt),
            "proximity_longest_run_frames": int(longest_run),
            "proximity_longest_run_seconds": float(longest_run * dt),
            "proximity_episodes": int(episodes),
        }
    return summary


def build_raw_arrays(frame_indices, timestamps, frame_shapes, detections_by_frame, metadata, track_summaries, selected_tracks):
    detections = [detection for frame in detections_by_frame for detection in frame]
    instance_count = len(detections)
    frame_offsets = [0]
    for frame in detections_by_frame:
        frame_offsets.append(frame_offsets[-1] + len(frame))

    mask_parts = []
    mask_offsets = [0]
    mask_origins = []
    mask_shapes = []
    polygons = []
    polygon_offsets = [0]

    for detection in detections:
        origin, shape, packed = pack_mask(detection["mask"])
        reconstructed = unpack_mask(packed, origin, shape, detection["mask"].shape)
        if not np.array_equal(detection["mask"], reconstructed):
            raise ValueError(
                "Falló round-trip de máscara antes de guardar instancia %d"
                % detection["global_index"]
            )
        mask_parts.append(packed)
        mask_offsets.append(mask_offsets[-1] + len(packed))
        mask_origins.append(origin)
        mask_shapes.append(shape)
        polygon = np.asarray(detection["polygon_xy"], dtype=np.float32).reshape(-1, 2)
        polygons.append(polygon)
        polygon_offsets.append(polygon_offsets[-1] + len(polygon))

    mask_bits = np.concatenate(mask_parts) if mask_parts else np.empty(0, dtype=np.uint8)
    polygon_xy = np.concatenate(polygons, axis=0) if polygons else np.empty((0, 2), dtype=np.float32)
    image_widths = np.asarray(frame_shapes)[:, 1].astype(np.float32)
    image_heights = np.asarray(frame_shapes)[:, 0].astype(np.float32)
    boxes = np.asarray([detection["bbox_xyxy"] for detection in detections], dtype=np.float32).reshape(instance_count, 4)
    box_xywh = np.column_stack((
        (boxes[:, 0] + boxes[:, 2]) / 2.0,
        (boxes[:, 1] + boxes[:, 3]) / 2.0,
        boxes[:, 2] - boxes[:, 0],
        boxes[:, 3] - boxes[:, 1],
    )) if instance_count else np.empty((0, 4), dtype=np.float32)
    detection_frames = np.asarray([detection["frame_index"] for detection in detections], dtype=np.int32)
    widths = image_widths[detection_frames] if instance_count else np.empty(0, dtype=np.float32)
    heights = image_heights[detection_frames] if instance_count else np.empty(0, dtype=np.float32)
    box_scale = np.column_stack((widths, heights, widths, heights)) if instance_count else np.empty((0, 4), dtype=np.float32)

    return {
        "schema_version": np.asarray(RAW_SCHEMA_VERSION),
        "metadata_json": np.asarray(json_dumps(metadata)),
        "track_summaries_json": np.asarray(json_dumps(track_summaries)),
        "selected_tracks_json": np.asarray(json_dumps(selected_tracks)),
        "frame_index": np.asarray(frame_indices, dtype=np.int32),
        "timestamp_sec": np.asarray(timestamps, dtype=np.float64),
        "frame_image_shape_hw": np.asarray(frame_shapes, dtype=np.int32),
        "frame_instance_offsets": np.asarray(frame_offsets, dtype=np.int64),
        "instance_frame_index": detection_frames,
        "instance_index_in_frame": np.asarray([detection["instance_index"] for detection in detections], dtype=np.int16),
        "instance_class_id": np.asarray([detection["class_id"] for detection in detections], dtype=np.int16),
        "instance_class_name": np.asarray([detection["class_name"] for detection in detections], dtype="<U32"),
        "instance_confidence": np.asarray([detection["confidence"] for detection in detections], dtype=np.float32),
        "instance_bbox_xyxy_px": boxes,
        "instance_bbox_xyxy_normalized": boxes / box_scale if instance_count else boxes.copy(),
        "instance_bbox_cxcywh_px": box_xywh,
        "instance_bbox_cxcywh_normalized": box_xywh / box_scale if instance_count else box_xywh.copy(),
        "instance_track_id": np.asarray([detection["track_id"] for detection in detections], dtype=np.int32),
        "instance_mask_origin_xy": np.asarray(mask_origins, dtype=np.int32).reshape(instance_count, 2),
        "instance_mask_shape_hw": np.asarray(mask_shapes, dtype=np.int32).reshape(instance_count, 2),
        "instance_mask_area_px": np.asarray([detection["geometry"]["area"] for detection in detections], dtype=np.int32),
        "mask_byte_offsets": np.asarray(mask_offsets, dtype=np.int64),
        "mask_bits": mask_bits,
        "polygon_point_offsets": np.asarray(polygon_offsets, dtype=np.int64),
        "polygon_xy_px": polygon_xy,
    }


def build_feature_arrays(frame_indices, timestamps, class_names, selected_tracks, object_features, selected_detection_indices, interaction_features, interaction_summary, metadata):
    class_ids = sorted(class_names)
    return {
        "schema_version": np.asarray(FEATURES_SCHEMA_VERSION),
        "metadata_json": np.asarray(json_dumps(metadata)),
        "interaction_summary_json": np.asarray(json_dumps(interaction_summary)),
        "frame_index": np.asarray(frame_indices, dtype=np.int32),
        "timestamp_sec": np.asarray(timestamps, dtype=np.float64),
        "class_ids": np.asarray(class_ids, dtype=np.int16),
        "class_names": np.asarray([class_names[class_id] for class_id in class_ids], dtype="<U32"),
        "selected_track_ids": np.asarray([selected_tracks.get(class_names[class_id], -1) for class_id in class_ids], dtype=np.int32),
        "object_feature_names": np.asarray(OBJECT_FEATURE_NAMES, dtype="<U64"),
        "object_features": np.asarray(object_features, dtype=np.float64),
        "selected_instance_indices": np.asarray(selected_detection_indices, dtype=np.int64),
        "interaction_pair_names": np.asarray(["_".join(pair) for pair in INTERACTION_PAIRS], dtype="<U32"),
        "interaction_feature_names": np.asarray(INTERACTION_FEATURE_NAMES, dtype="<U64"),
        "interaction_features": np.asarray(interaction_features, dtype=np.float64),
    }


def save_npz(path, arrays):
    path = Path(path)
    with path.open("wb") as file:
        np.savez_compressed(file, **arrays)


def validate_raw_file(path, original_masks=None):
    with np.load(path, allow_pickle=False) as data:
        required = {
            "schema_version", "metadata_json", "track_summaries_json",
            "selected_tracks_json", "frame_index", "timestamp_sec",
            "frame_image_shape_hw", "frame_instance_offsets", "instance_frame_index",
            "instance_index_in_frame", "instance_class_id", "instance_class_name",
            "instance_confidence", "instance_bbox_xyxy_px",
            "instance_bbox_xyxy_normalized", "instance_bbox_cxcywh_px",
            "instance_bbox_cxcywh_normalized", "instance_track_id", "instance_mask_origin_xy",
            "instance_mask_shape_hw", "instance_mask_area_px", "mask_byte_offsets",
            "mask_bits", "polygon_point_offsets", "polygon_xy_px",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError("Faltan arrays RAW: %s" % sorted(missing))
        schema_version = str(data["schema_version"])
        if schema_version not in {RAW_SCHEMA_VERSION, RETRACKED_RAW_SCHEMA_VERSION}:
            raise ValueError("Versión RAW inesperada")
        if schema_version == RETRACKED_RAW_SCHEMA_VERSION:
            retracked_required = {
                "instance_original_track_id", "original_track_summaries_json",
                "original_selected_tracks_json", "track_merge_audit_json",
                "source_raw_sha256",
            }
            retracked_missing = retracked_required - set(data.files)
            if retracked_missing:
                raise ValueError("Faltan arrays RAW retracked: %s" % sorted(retracked_missing))
        metadata = json.loads(str(data["metadata_json"]))
        required_metadata = {
            "extraction_id", "video_id", "video_path", "video_sha256",
            "video_size_bytes", "video_mtime_ns", "model_path", "model_sha256", "classes",
            "ultralytics_version", "opencv_version", "device_name", "imgsz",
            "confidence_threshold", "iou_threshold", "retina_masks",
            "proximity_threshold_normalized", "tracking", "mask_encoding",
            "fps", "width", "height", "frames_video", "frames_decoded",
            "frames_processed", "instances_total", "extraction_timestamp_utc",
        }
        missing_metadata = required_metadata - set(metadata)
        if missing_metadata:
            raise ValueError("Falta metadata RAW: %s" % sorted(missing_metadata))
        if metadata["retina_masks"] is not True:
            raise ValueError("retina_masks no está registrado como True")
        if schema_version == RETRACKED_RAW_SCHEMA_VERSION:
            derivation = metadata.get("derivation", {})
            if not derivation.get("derivation_id") or derivation.get("uses_yolo") is not False:
                raise ValueError("Metadata de derivación retracked inválida")
        frames_total = len(data["frame_index"])
        if not np.array_equal(data["frame_index"], np.arange(frames_total)):
            raise ValueError("Índices de frame no continuos")
        timestamps = data["timestamp_sec"]
        frame_shapes = data["frame_image_shape_hw"]
        if timestamps.shape != (frames_total,) or np.any(~np.isfinite(timestamps)):
            raise ValueError("Timestamps RAW inválidos")
        if frame_shapes.shape != (frames_total, 2) or np.any(frame_shapes <= 0):
            raise ValueError("Shapes de frames inválidos")
        if frames_total > 1 and np.any(np.diff(timestamps) <= 0):
            raise ValueError("Timestamps no estrictamente crecientes")
        frame_offsets = data["frame_instance_offsets"]
        if len(frame_offsets) != frames_total + 1:
            raise ValueError("Offsets por frame inconsistentes")
        if int(frame_offsets[0]) != 0 or np.any(np.diff(frame_offsets) < 0):
            raise ValueError("Offsets por frame no monótonos")
        instance_count = len(data["instance_frame_index"])
        if int(frame_offsets[-1]) != instance_count:
            raise ValueError("Cantidad de instancias inconsistente")
        instance_arrays = [
            "instance_index_in_frame", "instance_class_id", "instance_class_name",
            "instance_confidence", "instance_bbox_xyxy_px",
            "instance_bbox_xyxy_normalized", "instance_bbox_cxcywh_px",
            "instance_bbox_cxcywh_normalized", "instance_track_id",
            "instance_mask_origin_xy", "instance_mask_shape_hw", "instance_mask_area_px",
        ]
        if any(len(data[name]) != instance_count for name in instance_arrays):
            raise ValueError("Longitudes de arrays por instancia inconsistentes")
        if schema_version == RETRACKED_RAW_SCHEMA_VERSION:
            if len(data["instance_original_track_id"]) != instance_count:
                raise ValueError("Longitud de track IDs originales inconsistente")
            if np.any(data["instance_original_track_id"] < 0):
                raise ValueError("Track IDs originales negativos")
        mask_offsets = data["mask_byte_offsets"]
        if len(mask_offsets) != instance_count + 1:
            raise ValueError("Offsets de máscaras inconsistentes")
        if int(mask_offsets[0]) != 0 or np.any(np.diff(mask_offsets) < 0):
            raise ValueError("Offsets de máscaras no monótonos")
        if int(mask_offsets[-1]) != len(data["mask_bits"]):
            raise ValueError("Buffer de máscaras inconsistente")
        polygon_offsets = data["polygon_point_offsets"]
        if len(polygon_offsets) != instance_count + 1:
            raise ValueError("Offsets de polígonos inconsistentes")
        if int(polygon_offsets[0]) != 0 or np.any(np.diff(polygon_offsets) < 0):
            raise ValueError("Offsets de polígonos no monótonos")
        if int(polygon_offsets[-1]) != len(data["polygon_xy_px"]):
            raise ValueError("Buffer de polígonos inconsistente")
        if np.any(data["instance_track_id"] < 0):
            raise ValueError("Track IDs negativos en detecciones")
        numeric_detection_arrays = [
            "instance_confidence", "instance_bbox_xyxy_px",
            "instance_bbox_xyxy_normalized", "instance_bbox_cxcywh_px",
            "instance_bbox_cxcywh_normalized",
        ]
        if any(np.any(~np.isfinite(data[name])) for name in numeric_detection_arrays):
            raise ValueError("Inf o NaN en detecciones RAW")
        if np.any((data["instance_confidence"] < 0) | (data["instance_confidence"] > 1)):
            raise ValueError("Confidence RAW fuera de rango")
        if np.any(~np.isfinite(data["polygon_xy_px"])):
            raise ValueError("Polígonos contienen Inf o NaN")
        if original_masks is not None and len(original_masks) != instance_count:
            raise ValueError("Cantidad de máscaras originales inconsistente")

        class_map = {int(key): str(value) for key, value in metadata["classes"].items()}
        track_summaries = json.loads(str(data["track_summaries_json"]))
        selected_tracks = json.loads(str(data["selected_tracks_json"]))
        tracks_by_id = {int(track["track_id"]): track for track in track_summaries}
        if len(tracks_by_id) != len(track_summaries):
            raise ValueError("Track IDs duplicados en metadata")
        for frame_index in range(frames_total):
            start = int(frame_offsets[frame_index])
            end = int(frame_offsets[frame_index + 1])
            if np.any(data["instance_frame_index"][start:end] != frame_index):
                raise ValueError("Instancia asociada al frame incorrecto")
            if not np.array_equal(data["instance_index_in_frame"][start:end], np.arange(end - start)):
                raise ValueError("Índices locales de instancia inconsistentes")

        for index in range(instance_count):
            frame_index = int(data["instance_frame_index"][index])
            if frame_index < 0 or frame_index >= frames_total:
                raise ValueError("Frame inválido en instancia %d" % index)
            class_id = int(data["instance_class_id"][index])
            class_name = str(data["instance_class_name"][index])
            if class_map.get(class_id) != class_name:
                raise ValueError("Clase inconsistente en instancia %d" % index)
            track_id = int(data["instance_track_id"][index])
            track = tracks_by_id.get(track_id)
            if track is None or int(track["class_id"]) != class_id or str(track["class_name"]) != class_name:
                raise ValueError("Track inconsistente en instancia %d" % index)
            height, width = map(int, frame_shapes[frame_index])
            x1, y1, x2, y2 = data["instance_bbox_xyxy_px"][index]
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError("BBox fuera de imagen en instancia %d" % index)
            origin_x, origin_y = map(int, data["instance_mask_origin_xy"][index])
            crop_height, crop_width = map(int, data["instance_mask_shape_hw"][index])
            if not (crop_width > 0 and crop_height > 0):
                raise ValueError("Crop de máscara vacío en instancia %d" % index)
            if not (0 <= origin_x < width and 0 <= origin_y < height):
                raise ValueError("Origen de máscara fuera de imagen en instancia %d" % index)
            if origin_x + crop_width > width or origin_y + crop_height > height:
                raise ValueError("Crop de máscara fuera de imagen en instancia %d" % index)
            expected_bytes = (crop_height * crop_width + 7) // 8
            if int(mask_offsets[index + 1] - mask_offsets[index]) != expected_bytes:
                raise ValueError("Longitud packed inválida en instancia %d" % index)
            polygon = data["polygon_xy_px"][polygon_offsets[index]:polygon_offsets[index + 1]]
            if len(polygon) < 3:
                raise ValueError("Polígono degenerado en instancia %d" % index)
            if np.any(polygon[:, 0] < 0) or np.any(polygon[:, 0] > width):
                raise ValueError("Polígono X fuera de imagen en instancia %d" % index)
            if np.any(polygon[:, 1] < 0) or np.any(polygon[:, 1] > height):
                raise ValueError("Polígono Y fuera de imagen en instancia %d" % index)
            mask = reconstruct_mask(data, index)
            if int(np.count_nonzero(mask)) != int(data["instance_mask_area_px"][index]):
                raise ValueError("Área reconstruida distinta en instancia %d" % index)
            if original_masks is not None and not np.array_equal(mask, original_masks[index]):
                raise ValueError("Round-trip lossless falló en instancia %d" % index)
        if metadata.get("frames_processed") != frames_total:
            raise ValueError("Metadata frames_processed inconsistente")
        if metadata.get("instances_total") != instance_count:
            raise ValueError("Metadata instances_total inconsistente")
        if metadata.get("frames_decoded") != frames_total:
            raise ValueError("Metadata frames_decoded inconsistente")
        if metadata.get("width") != int(frame_shapes[0, 1]) or metadata.get("height") != int(frame_shapes[0, 0]):
            raise ValueError("Resolución en metadata inconsistente")
        for class_name, track_id in selected_tracks.items():
            track_id = int(track_id)
            if track_id >= 0:
                track = tracks_by_id.get(track_id)
                if track is None or track["class_name"] != class_name:
                    raise ValueError("Track principal inconsistente para %s" % class_name)
        for track_id, track in tracks_by_id.items():
            indices = np.flatnonzero(data["instance_track_id"] == track_id)
            if not len(indices):
                raise ValueError("Track %d sin detecciones" % track_id)
            frames = data["instance_frame_index"][indices]
            if len(frames) != len(np.unique(frames)):
                raise ValueError("Track %d contiene más de una instancia por frame" % track_id)
            expected = {
                "first_frame": int(frames.min()),
                "last_frame": int(frames.max()),
                "duration_frames": int(frames.max() - frames.min() + 1),
                "frames_covered": int(len(indices)),
            }
            if any(int(track.get(name, -1)) != value for name, value in expected.items()):
                raise ValueError("Resumen inconsistente para track %d" % track_id)
            mean_confidence = float(np.mean(data["instance_confidence"][indices]))
            if not np.isclose(float(track.get("mean_confidence", np.nan)), mean_confidence):
                raise ValueError("Confidence media inconsistente para track %d" % track_id)
        return {
            "schema_version": schema_version,
            "frames": frames_total,
            "instances": instance_count,
            "roundtrip_masks": instance_count,
            "metadata": metadata,
        }


def validate_features_file(path, raw_path=None):
    with np.load(path, allow_pickle=False) as data:
        required = {
            "schema_version", "metadata_json", "frame_index", "timestamp_sec",
            "class_ids", "class_names", "selected_track_ids", "object_feature_names",
            "object_features", "selected_instance_indices", "interaction_pair_names",
            "interaction_feature_names", "interaction_features",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError("Faltan arrays de features: %s" % sorted(missing))
        schema_version = str(data["schema_version"])
        if schema_version not in {FEATURES_SCHEMA_VERSION, RETRACKED_FEATURES_SCHEMA_VERSION}:
            raise ValueError("Versión de features inesperada")
        metadata = json.loads(str(data["metadata_json"]))
        if schema_version == RETRACKED_FEATURES_SCHEMA_VERSION:
            derivation = metadata.get("derivation", {})
            if not derivation.get("derivation_id") or derivation.get("uses_yolo") is not False:
                raise ValueError("Metadata de features retracked inválida")
        frames_total = len(data["frame_index"])
        if not np.array_equal(data["frame_index"], np.arange(frames_total)):
            raise ValueError("Índices de features no continuos")
        if data["timestamp_sec"].shape != (frames_total,) or np.any(~np.isfinite(data["timestamp_sec"])):
            raise ValueError("Timestamps de features inválidos")
        object_features = data["object_features"]
        interaction_features = data["interaction_features"]
        if data["object_feature_names"].astype(str).tolist() != OBJECT_FEATURE_NAMES:
            raise ValueError("Esquema de object_features inesperado")
        if data["interaction_feature_names"].astype(str).tolist() != INTERACTION_FEATURE_NAMES:
            raise ValueError("Esquema de interaction_features inesperado")
        expected_pairs = ["_".join(pair) for pair in INTERACTION_PAIRS]
        if data["interaction_pair_names"].astype(str).tolist() != expected_pairs:
            raise ValueError("Pares de interacción inesperados")
        if object_features.shape != (frames_total, len(data["class_ids"]), len(OBJECT_FEATURE_NAMES)):
            raise ValueError("Shape de object_features inconsistente")
        if interaction_features.shape != (frames_total, len(INTERACTION_PAIRS), len(INTERACTION_FEATURE_NAMES)):
            raise ValueError("Shape de interaction_features inconsistente")
        if np.any(np.isinf(object_features)) or np.any(np.isinf(interaction_features)):
            raise ValueError("Features contienen Inf")
        names = data["object_feature_names"].astype(str).tolist()
        detected_index = names.index("detected")
        confidence_index = names.index("confidence")
        detected = object_features[:, :, detected_index]
        if not np.all(np.isin(detected, [0.0, 1.0])):
            raise ValueError("Flag detected inválido")
        if np.any(~np.isfinite(object_features[:, :, confidence_index][detected == 1.0])):
            raise ValueError("Confidence ausente en objeto detectado")
        static_end = names.index("minor_axis_length_normalized") + 1
        static_values = object_features[:, :, 1:static_end]
        if np.any(~np.isfinite(static_values[detected == 1.0])):
            raise ValueError("NaN ilegítimo en feature geométrica detectada")
        if np.any(~np.isnan(static_values[detected == 0.0])):
            raise ValueError("Feature geométrica presente con detected=0")
        selected_instances = data["selected_instance_indices"]
        if selected_instances.shape != detected.shape:
            raise ValueError("Shape de selected_instance_indices inconsistente")
        if np.any((selected_instances >= 0) != (detected == 1.0)):
            raise ValueError("Flag detected e índice de instancia no coinciden")

        interaction_names = data["interaction_feature_names"].astype(str).tolist()
        both_index = interaction_names.index("both_detected")
        overlap_index = interaction_names.index("overlap_detected")
        proximity_index = interaction_names.index("proximity_candidate")
        both = interaction_features[:, :, both_index]
        if not np.all(np.isin(both, [0.0, 1.0])):
            raise ValueError("Flag both_detected inválido")
        continuous_indices = [
            index for index in range(len(interaction_names))
            if index not in {both_index, overlap_index, proximity_index}
        ]
        continuous = interaction_features[:, :, continuous_indices]
        if np.any(~np.isfinite(continuous[both == 1.0])):
            raise ValueError("NaN ilegítimo en interacción disponible")
        if np.any(~np.isnan(continuous[both == 0.0])):
            raise ValueError("Interacción continua presente sin ambas detecciones")
        if np.any(~np.isin(interaction_features[:, :, overlap_index], [0.0, 1.0])):
            raise ValueError("Flag overlap_detected inválido")
        if np.any(~np.isin(interaction_features[:, :, proximity_index], [0.0, 1.0])):
            raise ValueError("Flag proximity_candidate inválido")
        intersection_index = interaction_names.index("intersection_area_px")
        union_index = interaction_names.index("union_area_px")
        iou_index = interaction_names.index("iou")
        distance_index = interaction_names.index("minimum_mask_distance_normalized")
        available = both == 1.0
        intersections = interaction_features[:, :, intersection_index]
        unions = interaction_features[:, :, union_index]
        ious = interaction_features[:, :, iou_index]
        overlaps = interaction_features[:, :, overlap_index]
        proximities = interaction_features[:, :, proximity_index]
        distances = interaction_features[:, :, distance_index]
        if np.any(intersections[available] < 0) or np.any(unions[available] <= 0):
            raise ValueError("Áreas de interacción inválidas")
        if np.any((ious[available] < 0) | (ious[available] > 1)):
            raise ValueError("IoU fuera de rango")
        if np.any(overlaps[available] != (intersections[available] > 0)):
            raise ValueError("Flag overlap inconsistente")
        if np.any(distances[available & (overlaps == 1.0)] != 0.0):
            raise ValueError("Distancia no nula cuando existe overlap")
        expected_proximity = distances[available] <= float(metadata["proximity_threshold_normalized"])
        if np.any(proximities[available] != expected_proximity):
            raise ValueError("Flag proximity_candidate inconsistente")
        if raw_path is not None:
            with np.load(raw_path, allow_pickle=False) as raw:
                if len(raw["frame_index"]) != frames_total:
                    raise ValueError("RAW y features tienen distinto T")
                raw_metadata = json.loads(str(raw["metadata_json"]))
                paired_keys = [
                    "extraction_id", "video_id", "video_sha256", "model_sha256",
                    "imgsz", "confidence_threshold", "iou_threshold",
                    "proximity_threshold_normalized", "tracking", "classes",
                    "frames_processed", "instances_total",
                ]
                if any(metadata.get(key) != raw_metadata.get(key) for key in paired_keys):
                    raise ValueError("RAW y features no pertenecen a la misma extracción")
                if schema_version == RETRACKED_FEATURES_SCHEMA_VERSION:
                    raw_derivation = raw_metadata.get("derivation", {})
                    if metadata.get("derivation", {}).get("derivation_id") != raw_derivation.get("derivation_id"):
                        raise ValueError("RAW y features tienen distinta derivación")
                if not np.array_equal(data["frame_index"], raw["frame_index"]):
                    raise ValueError("Frames RAW/features inconsistentes")
                if not np.array_equal(data["timestamp_sec"], raw["timestamp_sec"]):
                    raise ValueError("Timestamps RAW/features inconsistentes")
                raw_class_ids = raw["instance_class_id"].astype(int)
                raw_class_names = raw["instance_class_name"].astype(str)
                class_ids = data["class_ids"].astype(int)
                class_names = data["class_names"].astype(str)
                expected_classes = [(int(key), value) for key, value in sorted(
                    ((int(key), value) for key, value in raw_metadata["classes"].items())
                )]
                if list(zip(class_ids.tolist(), class_names.tolist())) != expected_classes:
                    raise ValueError("Clases RAW/features inconsistentes")
                valid_tracks = set(raw["instance_track_id"].astype(int).tolist())
                selected = set(data["selected_track_ids"].astype(int).tolist()) - {-1}
                if not selected.issubset(valid_tracks):
                    raise ValueError("Track principal inexistente en RAW")
                valid_instances = selected_instances[selected_instances >= 0]
                if np.any(valid_instances >= len(raw["instance_frame_index"])):
                    raise ValueError("Índice de instancia principal inválido")
                for frame_index, class_position in np.argwhere(selected_instances >= 0):
                    instance_index = int(selected_instances[frame_index, class_position])
                    if int(raw["instance_frame_index"][instance_index]) != frame_index:
                        raise ValueError("Feature vinculada al frame RAW incorrecto")
                    if int(raw["instance_track_id"][instance_index]) != int(data["selected_track_ids"][class_position]):
                        raise ValueError("Feature vinculada al track incorrecto")
                    if raw_class_ids[instance_index] != class_ids[class_position] or raw_class_names[instance_index] != class_names[class_position]:
                        raise ValueError("Feature vinculada a la clase RAW incorrecta")
                    raw_index = object_features[
                        frame_index, class_position, names.index("raw_instance_index")
                    ]
                    if int(raw_index) != instance_index:
                        raise ValueError("Feature raw_instance_index inconsistente")
        return {"frames": frames_total, "metadata": metadata}


def atomic_publish(temp_path, final_path):
    os.replace(str(temp_path), str(final_path))


def atomic_publish_pair(raw_temp, raw_path, features_temp, features_path):
    raw_temp = Path(raw_temp)
    raw_path = Path(raw_path)
    features_temp = Path(features_temp)
    features_path = Path(features_path)
    raw_backup = raw_path.with_suffix(raw_path.suffix + ".backup")
    features_backup = features_path.with_suffix(features_path.suffix + ".backup")
    for backup in (raw_backup, features_backup):
        backup.unlink(missing_ok=True)

    moved_old_raw = False
    moved_old_features = False
    published_raw = False
    published_features = False
    try:
        if raw_path.exists():
            os.replace(str(raw_path), str(raw_backup))
            moved_old_raw = True
        if features_path.exists():
            os.replace(str(features_path), str(features_backup))
            moved_old_features = True
        os.replace(str(raw_temp), str(raw_path))
        published_raw = True
        os.replace(str(features_temp), str(features_path))
        published_features = True
    except Exception:
        if published_raw:
            raw_path.unlink(missing_ok=True)
        if published_features:
            features_path.unlink(missing_ok=True)
        if moved_old_raw:
            os.replace(str(raw_backup), str(raw_path))
        if moved_old_features:
            os.replace(str(features_backup), str(features_path))
        raise
    else:
        raw_backup.unlink(missing_ok=True)
        features_backup.unlink(missing_ok=True)


def run_self_test():
    mask = np.zeros((73, 101), dtype=bool)
    mask[7:40, 13:77] = True
    mask[20:25, 30:36] = False
    origin, shape, packed = pack_mask(mask)
    reconstructed = unpack_mask(packed, origin, shape, mask.shape)
    if not np.array_equal(mask, reconstructed):
        raise AssertionError("Round-trip sintético de máscara falló")

    mask_b = np.zeros_like(mask)
    mask_b[7:40, 80:90] = True
    distance, point_a, point_b = minimum_mask_distance(mask, mask_b, return_points=True)
    if not np.isclose(distance, 4.0):
        raise AssertionError("Distancia sintética inesperada: %s" % distance)
    drawn_distance = math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])
    if not np.isclose(distance, drawn_distance):
        raise AssertionError("Los extremos visuales no representan la distancia mínima")
    interaction = compute_mask_interaction(mask, mask_b, (44.0, 23.0), (84.5, 23.0), 0.01)
    if interaction["intersection_area_px"] != 0 or interaction["minimum_mask_distance_px"] <= 0:
        raise AssertionError("Interacción sintética sin overlap inválida")

    mask_b[:, :] = False
    mask_b[30:50, 60:90] = True
    interaction = compute_mask_interaction(mask, mask_b, (44.0, 23.0), (74.5, 39.5), 0.01)
    if interaction["intersection_area_px"] <= 0 or interaction["minimum_mask_distance_px"] != 0:
        raise AssertionError("Interacción sintética con overlap inválida")
    return True


if __name__ == "__main__":
    run_self_test()
    print("OK: pruebas sintéticas de máscaras e interacciones superadas")
