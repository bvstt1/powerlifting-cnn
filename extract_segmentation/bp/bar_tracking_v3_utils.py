import math

import numpy as np

from segmentation_utils import bbox_iou, mask_iou


ALGORITHM_VERSION = "bp_front_bar_tracking_v3_prototype_1"

DEFAULT_PARAMETERS = {
    "max_missing_gap_frames": 8,
    "max_overlap_frames": 3,
    "motion_window": 5,
    "max_track_center_x_delta_over_width": 0.18,
    "min_track_area_ratio": 0.25,
    "max_track_area_ratio": 4.0,
    "min_track_length_ratio": 0.35,
    "max_track_length_ratio": 2.8,
    "max_track_orientation_delta_deg": 40.0,
    "max_edge_x_delta_over_width": 0.10,
    "max_edge_y_delta_over_height": 0.35,
    "max_predicted_error_over_diagonal": 0.15,
    "minimum_edge_score": 0.58,
    "minimum_overlap_identity_evidence": 0.35,
    "handoff_penalty": 1.50,
    "gap_penalty_per_frame": 0.08,
    "minimum_coverage_gain": 3,
    "minimum_score_improvement": 0.75,
    "minimum_runner_up_margin": 0.25,
}

SCORE_WEIGHTS = {
    "temporal_score": 0.10,
    "centroid_distance_score": 0.15,
    "predicted_motion_score": 0.20,
    "orientation_score": 0.10,
    "length_score": 0.12,
    "area_score": 0.10,
    "bbox_geometry_score": 0.08,
    "confidence_score": 0.05,
    "mask_transition_score": 0.10,
}


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else None


def finite_std(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.std()) if values.size else None


def finite_median(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None


def safe_ratio(value_a, value_b):
    value_a = max(float(value_a), 1e-9)
    value_b = max(float(value_b), 1e-9)
    return float(max(value_a, value_b) / min(value_a, value_b))


def circular_angle_difference(angle_a, angle_b):
    difference = abs(float(angle_a) - float(angle_b)) % 180.0
    return float(min(difference, 180.0 - difference))


def circular_mean_std(angles):
    angles = np.asarray(angles, dtype=np.float64)
    angles = angles[np.isfinite(angles)]
    if not angles.size:
        return None, None
    radians = np.deg2rad(2.0 * angles)
    mean_vector = np.mean(np.exp(1j * radians))
    mean = math.degrees(math.atan2(mean_vector.imag, mean_vector.real)) / 2.0
    resultant = max(abs(mean_vector), 1e-12)
    std = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(resultant)))) / 2.0
    return float(mean), float(std)


def geometry_vector(detection):
    geometry = detection["geometry"]
    return np.asarray([
        geometry["centroid_x"], geometry["centroid_y"], geometry["area"],
        geometry["orientation_angle_deg"], geometry["major_axis_length"],
        geometry["minor_axis_length"],
    ], dtype=np.float64)


def robust_velocity(detections, side, window):
    ordered = sorted(detections, key=lambda detection: detection["frame_index"])
    values = ordered[-window:] if side == "end" else ordered[:window]
    if len(values) < 2:
        return np.zeros(2, dtype=np.float64)
    velocities = []
    for previous, current in zip(values, values[1:]):
        delta = current["frame_index"] - previous["frame_index"]
        if delta <= 0:
            continue
        previous_center = np.asarray([
            previous["geometry"]["centroid_x"], previous["geometry"]["centroid_y"]
        ])
        current_center = np.asarray([
            current["geometry"]["centroid_x"], current["geometry"]["centroid_y"]
        ])
        velocities.append((current_center - previous_center) / delta)
    return np.median(np.asarray(velocities), axis=0) if velocities else np.zeros(2)


def trajectory_points(detections, side, window):
    ordered = sorted(detections, key=lambda detection: detection["frame_index"])
    values = ordered[-window:] if side == "end" else ordered[:window]
    return [
        {
            "frame": int(detection["frame_index"]),
            "centroid_x": float(detection["geometry"]["centroid_x"]),
            "centroid_y": float(detection["geometry"]["centroid_y"]),
        }
        for detection in values
    ]


def build_tracklets(detections_by_frame, frames_total, image_shape, v2_main_track_id, parameters=None):
    parameters = parameters or DEFAULT_PARAMETERS
    grouped = {}
    for detections in detections_by_frame:
        for detection in detections:
            if detection["class_name"] != "bar":
                continue
            track_id = int(detection["v2_track_id"])
            grouped.setdefault(track_id, []).append(detection)

    if v2_main_track_id not in grouped:
        raise ValueError("El track BAR principal V2 no existe")

    height, width = map(float, image_shape)
    anchor_detections = grouped[v2_main_track_id]
    anchor_x = finite_median([d["geometry"]["centroid_x"] for d in anchor_detections])
    anchor_area = finite_median([d["geometry"]["area"] for d in anchor_detections])
    anchor_length = finite_median([d["geometry"]["major_axis_length"] for d in anchor_detections])
    anchor_angle = circular_mean_std([
        d["geometry"]["orientation_angle_deg"] for d in anchor_detections
    ])[0]

    tracklets = {}
    for track_id, detections in grouped.items():
        detections.sort(key=lambda detection: (detection["frame_index"], detection["global_index"]))
        frames = np.asarray([d["frame_index"] for d in detections], dtype=np.int32)
        centers = np.asarray([
            [d["geometry"]["centroid_x"], d["geometry"]["centroid_y"]]
            for d in detections
        ], dtype=np.float64)
        boxes = np.asarray([d["bbox_xyxy"] for d in detections], dtype=np.float64)
        areas = np.asarray([d["geometry"]["area"] for d in detections], dtype=np.float64)
        orientations = np.asarray([
            d["geometry"]["orientation_angle_deg"] for d in detections
        ], dtype=np.float64)
        major = np.asarray([
            d["geometry"]["major_axis_length"] for d in detections
        ], dtype=np.float64)
        minor = np.asarray([
            d["geometry"]["minor_axis_length"] for d in detections
        ], dtype=np.float64)
        confidences = np.asarray([d["confidence"] for d in detections], dtype=np.float64)
        orientation_mean, orientation_std = circular_mean_std(orientations)
        area_mean = finite_mean(areas)
        length_mean = finite_mean(major)
        center_x_median = finite_median(centers[:, 0])
        area_ratio = safe_ratio(area_mean, anchor_area)
        length_ratio = safe_ratio(length_mean, anchor_length)
        angle_delta = circular_angle_difference(orientation_mean, anchor_angle)
        rejection_reasons = []
        if abs(center_x_median - anchor_x) / width > parameters["max_track_center_x_delta_over_width"]:
            rejection_reasons.append("track_center_x_incompatible_with_v2_anchor")
        if not parameters["min_track_area_ratio"] <= area_mean / anchor_area <= parameters["max_track_area_ratio"]:
            rejection_reasons.append("track_area_incompatible_with_v2_anchor")
        if not parameters["min_track_length_ratio"] <= length_mean / anchor_length <= parameters["max_track_length_ratio"]:
            rejection_reasons.append("track_length_incompatible_with_v2_anchor")
        if angle_delta > parameters["max_track_orientation_delta_deg"]:
            rejection_reasons.append("track_orientation_incompatible_with_v2_anchor")

        tracklets[track_id] = {
            "track_id": int(track_id),
            "is_v2_main": bool(track_id == v2_main_track_id),
            "start_frame": int(frames.min()),
            "end_frame": int(frames.max()),
            "n_frames": int(len(frames)),
            "coverage": float(len(frames) / frames_total),
            "frames": frames.tolist(),
            "mean_confidence": float(confidences.mean()),
            "median_confidence": float(np.median(confidences)),
            "centroid_start": centers[0].tolist(),
            "centroid_end": centers[-1].tolist(),
            "centroid_x_median": center_x_median,
            "centroid_y_median": finite_median(centers[:, 1]),
            "bbox_start": boxes[0].tolist(),
            "bbox_end": boxes[-1].tolist(),
            "bbox_width_mean": finite_mean(boxes[:, 2] - boxes[:, 0]),
            "bbox_height_mean": finite_mean(boxes[:, 3] - boxes[:, 1]),
            "mask_area_mean": area_mean,
            "mask_area_std": finite_std(areas),
            "orientation_mean": orientation_mean,
            "orientation_std": orientation_std,
            "major_axis_length_mean": length_mean,
            "major_axis_length_std": finite_std(major),
            "minor_axis_length_mean": finite_mean(minor),
            "minor_axis_length_std": finite_std(minor),
            "end_velocity_px_per_frame": robust_velocity(
                detections, "end", parameters["motion_window"]
            ).tolist(),
            "start_velocity_px_per_frame": robust_velocity(
                detections, "start", parameters["motion_window"]
            ).tolist(),
            "recent_end_trajectory": trajectory_points(
                detections, "end", parameters["motion_window"]
            ),
            "initial_trajectory": trajectory_points(
                detections, "start", parameters["motion_window"]
            ),
            "anchor_area_ratio": area_ratio,
            "anchor_length_ratio": length_ratio,
            "anchor_orientation_delta_deg": angle_delta,
            "eligible": not rejection_reasons or track_id == v2_main_track_id,
            "eligibility_rejection_reasons": rejection_reasons,
            "detections": detections,
        }
    return tracklets


def gaussian_score(value, scale):
    return float(math.exp(-0.5 * (float(value) / max(float(scale), 1e-12)) ** 2))


def log_ratio_score(ratio, maximum):
    return float(max(0.0, 1.0 - min(math.log(max(ratio, 1.0)) / math.log(maximum), 1.0)))


def containment_ratio(detection_a, detection_b):
    intersection = int(np.count_nonzero(detection_a["mask"] & detection_b["mask"]))
    minimum_area = max(1, min(
        int(detection_a["geometry"]["area"]), int(detection_b["geometry"]["area"])
    ))
    return float(intersection / minimum_area)


def transition_metrics(source, target, transition_frame, image_shape, parameters):
    source_candidates = [
        detection for detection in source["detections"]
        if detection["frame_index"] < transition_frame
    ]
    target_candidates = [
        detection for detection in target["detections"]
        if detection["frame_index"] >= transition_frame
    ]
    if not source_candidates or not target_candidates:
        return None
    source_detection = source_candidates[-1]
    target_detection = target_candidates[0]
    frame_delta = int(target_detection["frame_index"] - source_detection["frame_index"])
    if frame_delta <= 0:
        return None
    gap_frames = frame_delta - 1
    if gap_frames > parameters["max_missing_gap_frames"]:
        return None

    height, width = map(float, image_shape)
    diagonal = math.hypot(width, height)
    center_a = np.asarray([
        source_detection["geometry"]["centroid_x"],
        source_detection["geometry"]["centroid_y"],
    ])
    center_b = np.asarray([
        target_detection["geometry"]["centroid_x"],
        target_detection["geometry"]["centroid_y"],
    ])
    observed = center_b - center_a
    forward_velocity = np.asarray(source["end_velocity_px_per_frame"])
    backward_velocity = np.asarray(target["start_velocity_px_per_frame"])
    forward_expected = center_a + forward_velocity * frame_delta
    backward_expected = center_b - backward_velocity * frame_delta
    forward_error = float(np.linalg.norm(center_b - forward_expected) / diagonal)
    backward_error = float(np.linalg.norm(center_a - backward_expected) / diagonal)
    predicted_error = min(forward_error, backward_error)
    area_ratio = safe_ratio(
        source_detection["geometry"]["area"], target_detection["geometry"]["area"]
    )
    length_ratio = safe_ratio(
        source_detection["geometry"]["major_axis_length"],
        target_detection["geometry"]["major_axis_length"],
    )
    orientation_delta = circular_angle_difference(
        source_detection["geometry"]["orientation_angle_deg"],
        target_detection["geometry"]["orientation_angle_deg"],
    )
    box_a = source_detection["bbox_xyxy"]
    box_b = target_detection["bbox_xyxy"]
    width_ratio = safe_ratio(box_a[2] - box_a[0], box_b[2] - box_b[0])
    height_ratio = safe_ratio(box_a[3] - box_a[1], box_b[3] - box_b[1])
    mask_overlap = mask_iou(source_detection["mask"], target_detection["mask"])
    box_overlap = bbox_iou(box_a, box_b)
    containment = containment_ratio(source_detection, target_detection)

    components = {
        "temporal_score": gaussian_score(gap_frames, 6.0),
        "centroid_distance_score": gaussian_score(np.linalg.norm(observed) / diagonal, 0.10),
        "predicted_motion_score": gaussian_score(predicted_error, 0.10),
        "orientation_score": gaussian_score(orientation_delta, 25.0),
        "length_score": log_ratio_score(length_ratio, 2.8),
        "area_score": log_ratio_score(area_ratio, 4.0),
        "bbox_geometry_score": 0.5 * (
            log_ratio_score(width_ratio, 2.8) + log_ratio_score(height_ratio, 4.0)
        ),
        "confidence_score": float(np.clip(
            (min(source_detection["confidence"], target_detection["confidence"]) - 0.25) / 0.65,
            0.0,
            1.0,
        )),
        "mask_transition_score": float(max(mask_overlap, box_overlap, 0.8 * containment)),
    }
    total_score = float(sum(
        SCORE_WEIGHTS[name] * value for name, value in components.items()
    ))
    return {
        "transition_frame": int(transition_frame),
        "source_instance_index": int(source_detection["global_index"]),
        "target_instance_index": int(target_detection["global_index"]),
        "source_frame": int(source_detection["frame_index"]),
        "target_frame": int(target_detection["frame_index"]),
        "frame_delta": frame_delta,
        "gap_frames": gap_frames,
        "centroid_dx_px": float(observed[0]),
        "centroid_dy_px": float(observed[1]),
        "centroid_dx_over_width": float(abs(observed[0]) / width),
        "centroid_dy_over_height": float(abs(observed[1]) / height),
        "centroid_distance_over_diagonal": float(np.linalg.norm(observed) / diagonal),
        "forward_predicted_position_error": forward_error,
        "backward_predicted_position_error": backward_error,
        "predicted_position_error": predicted_error,
        "orientation_delta_deg": orientation_delta,
        "major_axis_length_ratio": length_ratio,
        "area_ratio": area_ratio,
        "bbox_width_ratio": width_ratio,
        "bbox_height_ratio": height_ratio,
        "mask_iou": float(mask_overlap),
        "bbox_iou": float(box_overlap),
        "smaller_containment_ratio": containment,
        "source_confidence": float(source_detection["confidence"]),
        "target_confidence": float(target_detection["confidence"]),
        **components,
        "total_score": total_score,
    }


def evaluate_candidate(source, target, raw_detected_frames, image_shape, parameters=None):
    parameters = parameters or DEFAULT_PARAMETERS
    overlap_start = max(source["start_frame"], target["start_frame"])
    overlap_end = min(source["end_frame"], target["end_frame"])
    overlap_frames = max(0, overlap_end - overlap_start + 1)
    base = {
        "source_track": int(source["track_id"]),
        "target_track": int(target["track_id"]),
        "source_start_frame": int(source["start_frame"]),
        "source_end_frame": int(source["end_frame"]),
        "target_start_frame": int(target["start_frame"]),
        "target_end_frame": int(target["end_frame"]),
        "overlap_frames": int(overlap_frames),
    }
    reasons = []
    if not source["eligible"]:
        reasons.append("source_tracklet_ineligible")
    if not target["eligible"]:
        reasons.append("target_tracklet_ineligible")
    if target["start_frame"] <= source["start_frame"]:
        reasons.append("target_not_temporally_after_source")
    if target["end_frame"] <= source["end_frame"]:
        reasons.append("target_does_not_extend_source")
    if overlap_frames > parameters["max_overlap_frames"]:
        reasons.append("incompatible_temporal_overlap")
    if reasons:
        return {**base, "accepted": 0, "rejection_reason": "|".join(reasons), "total_score": 0.0}

    transition_start = max(target["start_frame"], source["start_frame"] + 1)
    transition_end = min(target["end_frame"], source["end_frame"] + parameters["max_missing_gap_frames"] + 1)
    evaluated = []
    for transition_frame in range(transition_start, transition_end + 1):
        metrics = transition_metrics(source, target, transition_frame, image_shape, parameters)
        if metrics is not None:
            metrics["retained_pair_frames"] = int(
                sum(detection["frame_index"] < transition_frame for detection in source["detections"])
                + sum(detection["frame_index"] >= transition_frame for detection in target["detections"])
            )
            evaluated.append(metrics)
    if not evaluated:
        return {**base, "accepted": 0, "rejection_reason": "no_valid_temporal_cut", "total_score": 0.0}
    best = max(evaluated, key=lambda row: (
        row["retained_pair_frames"], row["total_score"], -row["transition_frame"]
    ))

    by_frame_source = {detection["frame_index"]: detection for detection in source["detections"]}
    by_frame_target = {detection["frame_index"]: detection for detection in target["detections"]}
    shared_frames = sorted(set(by_frame_source) & set(by_frame_target))
    overlap_mask_ious = []
    overlap_bbox_ious = []
    overlap_containments = []
    for frame in shared_frames:
        detection_a = by_frame_source[frame]
        detection_b = by_frame_target[frame]
        overlap_mask_ious.append(mask_iou(detection_a["mask"], detection_b["mask"]))
        overlap_bbox_ious.append(bbox_iou(detection_a["bbox_xyxy"], detection_b["bbox_xyxy"]))
        overlap_containments.append(containment_ratio(detection_a, detection_b))
    best["overlap_mask_iou_max"] = max(overlap_mask_ious, default=0.0)
    best["overlap_bbox_iou_max"] = max(overlap_bbox_ious, default=0.0)
    best["overlap_containment_max"] = max(overlap_containments, default=0.0)

    hard_reasons = []
    if best["centroid_dx_over_width"] > parameters["max_edge_x_delta_over_width"]:
        hard_reasons.append("horizontal_jump_too_large")
    if best["centroid_dy_over_height"] > parameters["max_edge_y_delta_over_height"]:
        hard_reasons.append("vertical_jump_too_large")
    if best["predicted_position_error"] > parameters["max_predicted_error_over_diagonal"]:
        hard_reasons.append("predicted_motion_incompatible")
    if best["orientation_delta_deg"] > parameters["max_track_orientation_delta_deg"]:
        hard_reasons.append("orientation_incompatible")
    if best["major_axis_length_ratio"] > parameters["max_track_length_ratio"]:
        hard_reasons.append("length_incompatible")
    if best["area_ratio"] > parameters["max_track_area_ratio"]:
        hard_reasons.append("area_incompatible")
    if best["total_score"] < parameters["minimum_edge_score"]:
        hard_reasons.append("score_below_threshold")
    if overlap_frames:
        overlap_evidence = max(
            best["overlap_mask_iou_max"], best["overlap_bbox_iou_max"],
            best["overlap_containment_max"],
        )
        strong_kinematic_evidence = (
            best["predicted_position_error"] <= 0.04
            and best["centroid_dx_over_width"] <= 0.03
            and best["major_axis_length_ratio"] <= 1.50
            and best["area_ratio"] <= 1.75
        )
        if (
            overlap_evidence < parameters["minimum_overlap_identity_evidence"]
            and not strong_kinematic_evidence
        ):
            hard_reasons.append("overlap_without_same_identity_evidence")

    raw_gap_frames = list(range(best["source_frame"] + 1, best["target_frame"]))
    best["raw_gap_frames"] = raw_gap_frames
    best["raw_gap_without_detection"] = [
        frame for frame in raw_gap_frames if frame not in raw_detected_frames
    ]
    best["tracking_gap_with_raw_detection"] = [
        frame for frame in raw_gap_frames if frame in raw_detected_frames
    ]
    return {
        **base,
        **best,
        "accepted": int(not hard_reasons),
        "rejection_reason": "" if not hard_reasons else "|".join(hard_reasons),
    }


def build_candidates(tracklets, raw_detected_frames, image_shape, parameters=None):
    parameters = parameters or DEFAULT_PARAMETERS
    candidates = []
    ordered = sorted(tracklets.values(), key=lambda track: (track["start_frame"], track["track_id"]))
    for source in ordered:
        for target in ordered:
            if source["track_id"] == target["track_id"]:
                continue
            candidates.append(evaluate_candidate(
                source, target, raw_detected_frames, image_shape, parameters
            ))
    return candidates


def node_quality(detection, reference):
    geometry = detection["geometry"]
    confidence = float(np.clip((detection["confidence"] - 0.25) / 0.65, 0.0, 1.0))
    x_score = gaussian_score(
        abs(geometry["centroid_x"] - reference["centroid_x"]) / reference["width"], 0.08
    )
    area_score = gaussian_score(
        abs(math.log(max(geometry["area"], 1) / max(reference["area"], 1))), 0.70
    )
    length_score = gaussian_score(
        abs(math.log(max(geometry["major_axis_length"], 1e-6) / max(reference["length"], 1e-6))),
        0.50,
    )
    orientation_score = gaussian_score(
        circular_angle_difference(geometry["orientation_angle_deg"], reference["orientation"]),
        20.0,
    )
    quality = (
        0.20 * confidence + 0.25 * x_score + 0.20 * area_score
        + 0.20 * length_score + 0.15 * orientation_score
    )
    return float(quality)


def compose_path(path, edge_by_pair, tracklets):
    selected = {}
    accepted_edges = []
    for position, track_id in enumerate(path):
        start = -math.inf
        end = math.inf
        if position:
            incoming = edge_by_pair[(path[position - 1], track_id)]
            start = incoming["transition_frame"]
        if position + 1 < len(path):
            outgoing = edge_by_pair[(track_id, path[position + 1])]
            end = outgoing["transition_frame"] - 1
            accepted_edges.append(outgoing)
        for detection in tracklets[track_id]["detections"]:
            frame = int(detection["frame_index"])
            if start <= frame <= end:
                selected[frame] = detection
    return selected, accepted_edges


def path_score(path, selected, edges, tracklets, image_shape, parameters):
    height, width = map(float, image_shape)
    anchor = next(track for track in tracklets.values() if track["is_v2_main"])
    reference = {
        "centroid_x": anchor["centroid_x_median"],
        "area": anchor["mask_area_mean"],
        "length": anchor["major_axis_length_mean"],
        "orientation": anchor["orientation_mean"],
        "width": width,
    }
    node_reward = sum(1.0 + 0.35 * node_quality(detection, reference) for detection in selected.values())
    handoff_cost = parameters["handoff_penalty"] * len(edges)
    gap_cost = parameters["gap_penalty_per_frame"] * sum(edge["gap_frames"] for edge in edges)
    incompatibility_cost = 0.50 * sum(1.0 - edge["total_score"] for edge in edges)
    return {
        "node_reward": float(node_reward),
        "handoff_cost": float(handoff_cost),
        "gap_cost": float(gap_cost),
        "incompatibility_cost": float(incompatibility_cost),
        "total": float(node_reward - handoff_cost - gap_cost - incompatibility_cost),
    }


def enumerate_paths(tracklets, candidates, anchor_track_id):
    accepted = [candidate for candidate in candidates if candidate["accepted"]]
    adjacency = {}
    for candidate in accepted:
        adjacency.setdefault(candidate["source_track"], []).append(candidate["target_track"])
    for values in adjacency.values():
        values.sort()
    paths = []

    def visit(path):
        paths.append(tuple(path))
        for target in adjacency.get(path[-1], []):
            if target not in path:
                visit(path + [target])

    for track_id in sorted(tracklets):
        visit([track_id])
    unique = sorted(set(paths))
    return [path for path in unique if anchor_track_id in path]


def select_global_path(tracklets, candidates, v2_indices, raw_detected_frames, image_shape, parameters=None):
    parameters = parameters or DEFAULT_PARAMETERS
    anchor_track_id = next(
        track_id for track_id, track in tracklets.items() if track["is_v2_main"]
    )
    v2_selected = {
        frame: detection
        for track in tracklets.values()
        for detection in track["detections"]
        for frame in [int(detection["frame_index"])]
        if int(detection["global_index"]) == int(v2_indices[frame])
    }
    fast_path = len(v2_selected) == len(raw_detected_frames)
    baseline_score = path_score(
        (anchor_track_id,), v2_selected, [], tracklets, image_shape, parameters
    )
    if fast_path:
        return {
            "selected": v2_selected,
            "path": (anchor_track_id,),
            "accepted_edges": [],
            "published": False,
            "decision": "LOCKED_V2_COMPLETE_RAW_COVERAGE",
            "score": baseline_score,
            "v2_score": baseline_score,
            "runner_up_score": None,
            "score_margin": None,
            "coverage_gain": 0,
        }

    edge_by_pair = {
        (candidate["source_track"], candidate["target_track"]): candidate
        for candidate in candidates if candidate["accepted"]
    }
    evaluated = []
    for path in enumerate_paths(tracklets, candidates, anchor_track_id):
        selected, edges = compose_path(path, edge_by_pair, tracklets)
        score = path_score(path, selected, edges, tracklets, image_shape, parameters)
        evaluated.append({
            "path": path, "selected": selected, "edges": edges, "score": score,
            "coverage": len(selected), "handoffs": len(edges),
            "v2_frames_retained": len(set(selected) & set(v2_selected)),
            "gap_frames": sum(edge["gap_frames"] for edge in edges),
        })
    evaluated.sort(key=lambda item: (
        item["score"]["total"], item["coverage"], -item["handoffs"],
        item["v2_frames_retained"], -item["gap_frames"],
        tuple(-track_id for track_id in item["path"]),
    ), reverse=True)
    best = evaluated[0]
    runner_up = evaluated[1] if len(evaluated) > 1 else None
    score_improvement = best["score"]["total"] - baseline_score["total"]
    score_margin = (
        best["score"]["total"] - runner_up["score"]["total"]
        if runner_up is not None else math.inf
    )
    coverage_gain = len(best["selected"]) - len(v2_selected)
    publish_reasons = []
    if coverage_gain < parameters["minimum_coverage_gain"]:
        publish_reasons.append("coverage_gain_below_minimum")
    if score_improvement < parameters["minimum_score_improvement"]:
        publish_reasons.append("score_improvement_below_minimum")
    if score_margin < parameters["minimum_runner_up_margin"]:
        publish_reasons.append("runner_up_margin_below_minimum")
    if len(best["selected"]) < len(v2_selected):
        publish_reasons.append("coverage_regression")
    published = not publish_reasons and tuple(best["path"]) != (anchor_track_id,)
    if not published:
        return {
            "selected": v2_selected,
            "path": (anchor_track_id,),
            "accepted_edges": [],
            "proposed_path": best["path"],
            "proposed_edges": best["edges"],
            "published": False,
            "decision": "KEEP_V2:" + "|".join(publish_reasons or ["v2_path_is_optimal"]),
            "score": baseline_score,
            "proposed_score": best["score"],
            "v2_score": baseline_score,
            "runner_up_score": runner_up["score"] if runner_up else None,
            "score_margin": float(score_margin) if np.isfinite(score_margin) else None,
            "coverage_gain": 0,
        }
    return {
        "selected": best["selected"],
        "path": best["path"],
        "accepted_edges": best["edges"],
        "published": True,
        "decision": "PUBLISH_GLOBAL_PATH",
        "score": best["score"],
        "v2_score": baseline_score,
        "runner_up_score": runner_up["score"] if runner_up else None,
        "score_margin": float(score_margin) if np.isfinite(score_margin) else None,
        "coverage_gain": int(coverage_gain),
    }


def serializable_tracklet(tracklet):
    return {key: value for key, value in tracklet.items() if key != "detections"}
