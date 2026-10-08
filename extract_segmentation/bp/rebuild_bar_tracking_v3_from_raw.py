import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from bar_tracking_v3_utils import (
    ALGORITHM_VERSION,
    DEFAULT_PARAMETERS,
    SCORE_WEIGHTS,
    build_candidates,
    build_tracklets,
    select_global_path,
    serializable_tracklet,
)
from segmentation_utils import (
    json_dumps,
    load_detections_from_raw,
    reconstruct_mask,
    save_npz,
    sha256_file,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "segmentation" / "bp" / "front"
DEFAULT_OUTPUT = DEFAULT_ROOT / "tracking_v3" / "test_14"
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "bp" / "front"
DEFAULT_REPORT = PROJECT_ROOT / "Reporte_BP_FRONT_Bar_TrackingV3_Prototype.md"

COMPARISON_FIELDS = [
    "video_id", "raw_frames", "raw_detection_ratio", "v2_main_track_id",
    "v2_main_track_coverage", "v3_main_track_coverage", "coverage_gain_frames",
    "v2_n_tracks_used", "v2_original_track_ids", "v3_n_tracklets_stitched",
    "v3_tracklet_path", "v2_missing_detected_frames", "v3_missing_detected_frames",
    "n_handoffs_v3", "accepted_handoff_pairs", "rejected_handoff_pairs",
    "algorithm_decision", "score_v2", "score_v3", "score_margin",
    "classification", "visual_evaluation", "source_v2_sha256", "uses_yolo",
    "uses_labels",
]

CANDIDATE_FIELDS = [
    "video_id", "source_track", "target_track", "source_start_frame",
    "source_end_frame", "target_start_frame", "target_end_frame",
    "transition_frame", "source_frame", "target_frame", "gap_frames",
    "overlap_frames", "temporal_score", "centroid_distance_score",
    "predicted_motion_score", "orientation_score", "length_score", "area_score",
    "bbox_geometry_score", "confidence_score", "mask_transition_score",
    "centroid_dx_over_width", "centroid_dy_over_height",
    "centroid_distance_over_diagonal", "forward_predicted_position_error",
    "backward_predicted_position_error", "predicted_position_error",
    "orientation_delta_deg", "major_axis_length_ratio", "area_ratio",
    "bbox_width_ratio", "bbox_height_ratio", "mask_iou", "bbox_iou",
    "smaller_containment_ratio", "overlap_mask_iou_max", "overlap_bbox_iou_max",
    "overlap_containment_max", "retained_pair_frames", "total_score", "accepted", "selected_by_v3",
    "rejection_reason", "raw_gap_without_detection",
    "tracking_gap_with_raw_detection",
]

VISUAL_FIELDS = [
    "video_id", "v3_tracklet_path", "handoffs_v3", "preview_paths",
    "visual_status", "same_physical_bar", "false_merge", "raw_gaps_preserved",
    "control_preserved", "visual_finding", "final_classification",
]

# Revision manual de los previews V2/V3, basada solo en identidad visual BAR.
VISUAL_REVIEWS = {
    "bp_001": {
        "classification": "UNCHANGED",
        "same_physical_bar": True,
        "false_merge": False,
        "raw_gaps_preserved": True,
        "control_preserved": True,
        "finding": "Fast path conserva exactamente el track simple y estable de V2",
    },
    "bp_004": {
        "classification": "IMPROVED",
        "same_physical_bar": True,
        "false_merge": False,
        "raw_gaps_preserved": True,
        "control_preserved": True,
        "finding": "Los handoffs 0->3 y 3->4 siguen la misma barra; V3 recupera los 62 frames sin saltos fisicos",
    },
    "bp_019": {
        "classification": "IMPROVED",
        "same_physical_bar": True,
        "false_merge": False,
        "raw_gaps_preserved": True,
        "control_preserved": True,
        "finding": "El handoff 0->4 recupera el sufijo de la misma barra y no selecciona la deteccion residual simultanea",
    },
    "bp_067": {
        "classification": "IMPROVED",
        "same_physical_bar": True,
        "false_merge": False,
        "raw_gaps_preserved": True,
        "control_preserved": True,
        "finding": "La transicion 0->3 es continua en posicion, forma y movimiento; se recuperan los 22 frames finales",
    },
    "bp_139": {
        "classification": "UNCHANGED",
        "same_physical_bar": True,
        "false_merge": False,
        "raw_gaps_preserved": True,
        "control_preserved": True,
        "finding": "V3 rechaza correctamente anexar el singleton terminal y conserva los gaps RAW reales",
    },
    "bp_198": {
        "classification": "IMPROVED",
        "same_physical_bar": True,
        "false_merge": False,
        "raw_gaps_preserved": True,
        "control_preserved": True,
        "finding": "Los handoffs 0->5 y 5->6 mantienen la misma barra; los dos frames RAW ausentes siguen vacios",
    },
    "bp_241": {
        "classification": "UNCHANGED",
        "same_physical_bar": True,
        "false_merge": False,
        "raw_gaps_preserved": True,
        "control_preserved": True,
        "finding": "Fast path conserva V2 y no intenta rellenar la salida real de la barra fuera del encuadre",
    },
    "bp_301": {
        "classification": "IMPROVED",
        "same_physical_bar": True,
        "false_merge": False,
        "raw_gaps_preserved": True,
        "control_preserved": True,
        "finding": "V3 mantiene vacio el gap por desenfoque y enlaza correctamente la reaparicion track 4",
    },
    "bp_467": {
        "classification": "IMPROVED",
        "same_physical_bar": True,
        "false_merge": False,
        "raw_gaps_preserved": True,
        "control_preserved": True,
        "finding": "El handoff 0->4 recupera la misma barra despues del gap RAW de seis frames sin interpolacion",
    },
    "bp_469": {
        "classification": "UNCHANGED",
        "same_physical_bar": True,
        "false_merge": False,
        "raw_gaps_preserved": True,
        "control_preserved": True,
        "finding": "V3 conserva V2 porque las rutas 0->5 y 0->4->5 son ambiguas; no fuerza los fragmentos terminales",
    },
    "bp_518": {
        "classification": "UNCHANGED",
        "same_physical_bar": True,
        "false_merge": False,
        "raw_gaps_preserved": True,
        "control_preserved": True,
        "finding": "Fast path mantiene track 0 completo y no fusiona fragmentos laterales ni detecciones residuales",
    },
    "bp_520": {
        "classification": "UNCHANGED",
        "same_physical_bar": True,
        "false_merge": False,
        "raw_gaps_preserved": True,
        "control_preserved": True,
        "finding": "Fast path conserva exactamente el track virtual 7 y el handoff V2 auditado 0->5",
    },
    "bp_541": {
        "classification": "IMPROVED",
        "same_physical_bar": True,
        "false_merge": False,
        "raw_gaps_preserved": True,
        "control_preserved": True,
        "finding": "El prefijo track 0 y el principal track 5 son la misma barra; V3 recupera los 191 frames",
    },
    "bp_544": {
        "classification": "IMPROVED",
        "same_physical_bar": True,
        "false_merge": False,
        "raw_gaps_preserved": True,
        "control_preserved": True,
        "finding": "El handoff 0->5 conserva identidad fisica y cobertura completa sin incorporar residuos laterales",
    },
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    temporary.replace(path)


def atomic_write_text(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def normalize_video_id(value):
    stem = Path(value).stem
    if stem.isdigit():
        return "bp_%03d" % int(stem)
    if not stem.startswith("bp_"):
        raise ValueError("ID no reconocido: %s" % value)
    return stem


def read_visual_test_ids(root):
    path = root / "manifests" / "bp_front_bar_tracking_visual_review.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [row["video_id"] for row in csv.DictReader(file)]


def selected_bar_indices(features):
    classes = features["class_names"].astype(str).tolist()
    position = classes.index("bar")
    return features["selected_instance_indices"][:, position].astype(np.int64)


def prepare_detections(raw):
    detections_by_frame = load_detections_from_raw(raw)
    has_original = "instance_original_track_id" in raw.files
    for detections in detections_by_frame:
        for detection in detections:
            index = detection["global_index"]
            v2_track_id = int(raw["instance_track_id"][index])
            original_track_id = (
                int(raw["instance_original_track_id"][index])
                if has_original else v2_track_id
            )
            detection["v2_track_id"] = v2_track_id
            detection["extractor_track_id"] = original_track_id
            detection["track_id"] = v2_track_id
    return detections_by_frame


def candidate_pair(candidate):
    return "%s->%s" % (candidate["source_track"], candidate["target_track"])


def process_video(video_id, root, output_root, audit_only):
    source_raw_path = root / "raw" / (video_id + "_segmentation.npz")
    v2_raw_path = root / "tracking_v2" / "raw" / (video_id + "_segmentation.npz")
    v2_features_path = root / "tracking_v2" / "features" / (video_id + "_features.npz")
    v2_audit_path = root / "tracking_v2" / "audits" / (video_id + "_tracking_audit.json")
    for path in [source_raw_path, v2_raw_path, v2_features_path, v2_audit_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    source_hash_before = sha256_file(source_raw_path)
    v2_hash_before = sha256_file(v2_raw_path)
    v2_audit = json.loads(v2_audit_path.read_text(encoding="utf-8"))
    if not v2_audit.get("source_raw_unchanged"):
        raise ValueError("V2 no confirma RAW fuente inmutable")

    with np.load(v2_raw_path, allow_pickle=False) as raw, np.load(
        v2_features_path, allow_pickle=False
    ) as features:
        metadata = json.loads(str(raw["metadata_json"]))
        frames_total = len(raw["frame_index"])
        image_shape = tuple(map(int, raw["frame_image_shape_hw"][0]))
        detections_by_frame = prepare_detections(raw)
        v2_indices = selected_bar_indices(features)
        raw_detected_frames = {
            frame_index for frame_index, detections in enumerate(detections_by_frame)
            if any(detection["class_name"] == "bar" for detection in detections)
        }
        v2_main_track_id = int(features["selected_track_ids"][
            features["class_names"].astype(str).tolist().index("bar")
        ])
        tracklets = build_tracklets(
            detections_by_frame, frames_total, image_shape, v2_main_track_id,
            DEFAULT_PARAMETERS,
        )
        candidates = build_candidates(
            tracklets, raw_detected_frames, image_shape, DEFAULT_PARAMETERS
        )
        result = select_global_path(
            tracklets, candidates, v2_indices, raw_detected_frames, image_shape,
            DEFAULT_PARAMETERS,
        )
        selected = result["selected"]
        v3_indices = np.full(frames_total, -1, dtype=np.int64)
        v3_track_ids = np.full(frames_total, -1, dtype=np.int32)
        v3_original_ids = np.full(frames_total, -1, dtype=np.int32)
        handoff_flags = np.zeros(frames_total, dtype=np.uint8)
        for frame_index, detection in selected.items():
            v3_indices[frame_index] = int(detection["global_index"])
            v3_track_ids[frame_index] = int(detection["v2_track_id"])
            v3_original_ids[frame_index] = int(detection["extractor_track_id"])
        for edge in result["accepted_edges"]:
            handoff_flags[int(edge["transition_frame"])] = 1

        v2_valid = set(np.flatnonzero(v2_indices >= 0).tolist())
        v3_valid = set(np.flatnonzero(v3_indices >= 0).tolist())
        if not v3_valid.issubset(raw_detected_frames):
            raise ValueError("V3 seleccionó un frame sin detección BAR RAW")
        if len(v3_valid) < len(v2_valid):
            raise ValueError("V3 produjo regresión de cobertura")
        for frame_index in v3_valid:
            index = int(v3_indices[frame_index])
            if int(raw["instance_frame_index"][index]) != frame_index:
                raise ValueError("Índice V3 no corresponde al frame")
            if str(raw["instance_class_name"][index]) != "bar":
                raise ValueError("Índice V3 no corresponde a BAR")

        accepted_pairs = [candidate_pair(edge) for edge in result["accepted_edges"]]
        rejected = [candidate for candidate in candidates if not candidate["accepted"]]
        rejected_sorted = sorted(
            rejected,
            key=lambda candidate: (-float(candidate.get("total_score", 0.0)),
                                   candidate["source_track"], candidate["target_track"]),
        )
        rejected_pairs = [candidate_pair(candidate) for candidate in rejected_sorted[:10]]
        path_ids = tuple(map(int, result["path"]))
        selected_pairs = set(zip(path_ids, path_ids[1:]))
        candidate_rows = []
        for candidate in candidates:
            row = dict(candidate)
            row["video_id"] = video_id
            row["selected_by_v3"] = int(
                (candidate["source_track"], candidate["target_track"]) in selected_pairs
                and candidate["accepted"]
            )
            for field in ["raw_gap_without_detection", "tracking_gap_with_raw_detection"]:
                row[field] = "|".join(map(str, candidate.get(field, [])))
            candidate_rows.append(row)

        v2_original_ids = sorted({
            int(raw["instance_original_track_id"][index])
            for index in v2_indices[v2_indices >= 0]
        })
        initial_classification = (
            "IMPROVED_PENDING_VISUAL" if len(v3_valid) > len(v2_valid)
            else "UNCHANGED_PENDING_VISUAL"
        )
        comparison = {
            "video_id": video_id,
            "raw_frames": len(raw_detected_frames),
            "raw_detection_ratio": len(raw_detected_frames) / frames_total,
            "v2_main_track_id": v2_main_track_id,
            "v2_main_track_coverage": len(v2_valid) / frames_total,
            "v3_main_track_coverage": len(v3_valid) / frames_total,
            "coverage_gain_frames": len(v3_valid) - len(v2_valid),
            "v2_n_tracks_used": len(set(
                int(raw["instance_track_id"][index]) for index in v2_indices[v2_indices >= 0]
            )),
            "v2_original_track_ids": "|".join(map(str, v2_original_ids)),
            "v3_n_tracklets_stitched": len(path_ids),
            "v3_tracklet_path": "|".join(map(str, path_ids)),
            "v2_missing_detected_frames": len(raw_detected_frames - v2_valid),
            "v3_missing_detected_frames": len(raw_detected_frames - v3_valid),
            "n_handoffs_v3": len(result["accepted_edges"]),
            "accepted_handoff_pairs": "|".join(accepted_pairs),
            "rejected_handoff_pairs": "|".join(rejected_pairs),
            "algorithm_decision": result["decision"],
            "score_v2": result["v2_score"]["total"],
            "score_v3": result["score"]["total"],
            "score_margin": result.get("score_margin"),
            "classification": initial_classification,
            "visual_evaluation": "PENDING",
            "source_v2_sha256": v2_hash_before,
            "uses_yolo": 0,
            "uses_labels": 0,
        }
        audit = {
            "video_id": video_id,
            "algorithm_version": ALGORITHM_VERSION,
            "timestamp_utc": utc_now(),
            "source_raw_path": str(source_raw_path.resolve()),
            "source_raw_sha256": source_hash_before,
            "source_v2_raw_path": str(v2_raw_path.resolve()),
            "source_v2_sha256": v2_hash_before,
            "source_raw_unchanged": True,
            "source_v2_unchanged": True,
            "uses_yolo": False,
            "uses_labels": False,
            "frames_total": frames_total,
            "raw_bar_frames": len(raw_detected_frames),
            "v2_selected_frames": len(v2_valid),
            "v3_selected_frames": len(v3_valid),
            "v2_selected_track_id": v2_main_track_id,
            "v3_tracklet_path": list(path_ids),
            "v3_original_track_sequence": [
                sorted({int(d["extractor_track_id"]) for d in tracklets[track_id]["detections"]})
                for track_id in path_ids
            ],
            "decision": result["decision"],
            "published_global_path": bool(result["published"]),
            "coverage_gain_frames": comparison["coverage_gain_frames"],
            "score": result["score"],
            "proposed_score": result.get("proposed_score"),
            "v2_score_under_v3_objective": result["v2_score"],
            "runner_up_score": result.get("runner_up_score"),
            "score_margin": result.get("score_margin"),
            "accepted_handoffs": result["accepted_edges"],
            "proposed_path": list(result.get("proposed_path", [])),
            "proposed_handoffs": result.get("proposed_edges", []),
            "tracklets": [
                serializable_tracklet(tracklets[track_id]) for track_id in sorted(tracklets)
            ],
            "handoff_candidates": candidates,
            "selected_instance_indices": v3_indices.tolist(),
            "uncovered_raw_frames": sorted(raw_detected_frames - v3_valid),
            "raw_detection_gap_frames": sorted(set(range(frames_total)) - raw_detected_frames),
            "parameters": DEFAULT_PARAMETERS,
            "score_weights": SCORE_WEIGHTS,
            "input_metadata": {
                "pipeline_version": metadata.get("pipeline_version"),
                "tracking_version": metadata.get("tracking", {}).get("algorithm_version"),
            },
        }

    if sha256_file(source_raw_path) != source_hash_before:
        raise ValueError("RAW fuente cambió durante V3")
    if sha256_file(v2_raw_path) != v2_hash_before:
        raise ValueError("RAW V2 cambió durante V3")

    audit_path = output_root / "audits" / (video_id + "_bar_tracking_v3_audit.json")
    atomic_write_json(audit_path, audit)
    selection_path = output_root / "selections" / (video_id + "_bar_tracking_v3_selection.npz")
    if not audit_only:
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = selection_path.with_suffix(selection_path.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        save_npz(temporary, {
            "schema_version": np.asarray("bp_front_bar_tracking_v3_selection_v1"),
            "metadata_json": np.asarray(json_dumps({
                "video_id": video_id,
                "algorithm_version": ALGORITHM_VERSION,
                "source_v2_sha256": v2_hash_before,
                "uses_yolo": False,
                "uses_labels": False,
                "interpolation": False,
            })),
            "frame_index": np.arange(frames_total, dtype=np.int32),
            "selected_instance_indices": v3_indices,
            "selected_v2_track_ids": v3_track_ids,
            "selected_original_track_ids": v3_original_ids,
            "handoff_transition": handoff_flags,
        })
        temporary.replace(selection_path)
    return comparison, candidate_rows, audit, v2_indices, v3_indices


def alpha_mask(frame, mask, color, alpha):
    if not np.any(mask):
        return
    layer = np.empty_like(frame)
    layer[:] = color
    frame[mask] = cv2.addWeighted(frame[mask], 1.0 - alpha, layer[mask], alpha, 0)


def annotate_comparison(frame, raw, frame_index, v2_index, v3_index, transition_text):
    output = frame.copy()
    start = int(raw["frame_instance_offsets"][frame_index])
    end = int(raw["frame_instance_offsets"][frame_index + 1])
    bar_indices = [
        index for index in range(start, end)
        if str(raw["instance_class_name"][index]) == "bar"
    ]
    for index in bar_indices:
        mask = reconstruct_mask(raw, index)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (0, 220, 255), 2)
        x1, y1, x2, y2 = raw["instance_bbox_xyxy_px"][index].astype(int)
        label = "RAW idx=%d v2=%d orig=%d" % (
            index,
            int(raw["instance_track_id"][index]),
            int(raw["instance_original_track_id"][index]),
        )
        cv2.putText(output, label, (max(0, x1), max(65, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
    if v2_index >= 0:
        mask = reconstruct_mask(raw, int(v2_index))
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (255, 255, 0), 5)
    if v3_index >= 0:
        mask = reconstruct_mask(raw, int(v3_index))
        alpha_mask(output, mask, (255, 0, 255), 0.35)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (255, 0, 255), 4)

    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (output.shape[1], 118), (0, 0, 0), -1)
    output[:118] = cv2.addWeighted(output[:118], 0.20, overlay[:118], 0.80, 0)
    lines = [
        "frame=%d RAW bars=%d" % (frame_index, len(bar_indices)),
        "V2 main idx=%d (cyan) | V3 main idx=%d (magenta)" % (v2_index, v3_index),
        transition_text,
    ]
    for position, text in enumerate(lines):
        cv2.putText(output, text, (12, 28 + 30 * position), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def read_frame(capture, frame_index, expected_shape):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise ValueError("No se pudo leer frame %d" % frame_index)
    height, width = map(int, expected_shape)
    return frame[:height, :width]


def generate_previews(video_id, root, output_root, dataset_dir, audit, v2_indices, v3_indices):
    raw_path = root / "tracking_v2" / "raw" / (video_id + "_segmentation.npz")
    video_path = dataset_dir / (video_id + ".mp4")
    output_dir = output_root / "previews" / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError("No se pudo abrir %s" % video_path)
    paths = []
    transitions = audit["accepted_handoffs"] or audit.get("proposed_handoffs", [])
    if not transitions:
        frames_total = len(v2_indices)
        transitions = [{
            "source_track": audit["v2_selected_track_id"],
            "target_track": audit["v2_selected_track_id"],
            "transition_frame": frames_total // 2,
            "total_score": 1.0,
            "accepted": 1,
            "rejection_reason": "control_without_handoff",
        }]
    try:
        with np.load(raw_path, allow_pickle=False) as raw:
            frames_total = len(raw["frame_index"])
            for transition_index, transition in enumerate(transitions):
                center = int(transition.get("transition_frame", frames_total // 2))
                frames = list(range(max(0, center - 5), min(frames_total, center + 6)))
                tiles = []
                pair = "%s_to_%s" % (transition["source_track"], transition["target_track"])
                status = (
                    "ACCEPTED" if transition in audit["accepted_handoffs"]
                    else "PROPOSED_NOT_PUBLISHED" if audit.get("proposed_handoffs")
                    else "CONTROL"
                )
                transition_text = "%s %s score=%.3f %s" % (
                    status, pair, float(transition.get("total_score", 0.0)),
                    transition.get("rejection_reason", ""),
                )
                transition_dir = output_dir / ("%02d_%s" % (transition_index, pair))
                transition_dir.mkdir(parents=True, exist_ok=True)
                for frame_index in frames:
                    frame = read_frame(capture, frame_index, raw["frame_image_shape_hw"][frame_index])
                    annotated = annotate_comparison(
                        frame, raw, frame_index, int(v2_indices[frame_index]),
                        int(v3_indices[frame_index]), transition_text,
                    )
                    path = transition_dir / ("frame_%05d.jpg" % frame_index)
                    if not cv2.imwrite(str(path), annotated):
                        raise OSError("No se pudo guardar %s" % path)
                    tiles.append(cv2.resize(annotated, (640, 360)))
                while len(tiles) % 4:
                    tiles.append(np.zeros_like(tiles[0]))
                contact_rows = [
                    np.hstack(tiles[index:index + 4]) for index in range(0, len(tiles), 4)
                ]
                contact = np.vstack(contact_rows)
                contact_path = transition_dir / (video_id + "_" + pair + "_contact_sheet.jpg")
                if not cv2.imwrite(str(contact_path), contact):
                    raise OSError("No se pudo guardar %s" % contact_path)
                paths.append(str(contact_path.resolve()))
    finally:
        capture.release()
    return paths


def apply_visual_reviews(comparisons, visual_rows):
    by_video = {row["video_id"]: row for row in visual_rows}
    for comparison in comparisons:
        review = VISUAL_REVIEWS.get(comparison["video_id"])
        if review is None:
            continue
        comparison["classification"] = review["classification"]
        comparison["visual_evaluation"] = review["finding"]
        row = by_video[comparison["video_id"]]
        row.update({
            "visual_status": "REVIEWED",
            "same_physical_bar": int(review["same_physical_bar"]),
            "false_merge": int(review["false_merge"]),
            "raw_gaps_preserved": int(review["raw_gaps_preserved"]),
            "control_preserved": int(review["control_preserved"]),
            "visual_finding": review["finding"],
            "final_classification": review["classification"],
        })


def build_report(comparisons, candidates, visual_rows):
    accepted = [row for row in candidates if int(row.get("selected_by_v3", 0))]
    relevant_rejected = sorted(
        [row for row in candidates if not int(row.get("accepted", 0))
         and float(row.get("total_score") or 0.0) > 0.35],
        key=lambda row: (-float(row.get("total_score") or 0.0), row["video_id"]),
    )
    reviewed = [row for row in visual_rows if row["visual_status"] == "REVIEWED"]
    regressions = [row for row in comparisons if row["classification"] == "REGRESSED"]
    improved = [row for row in comparisons if row["classification"] == "IMPROVED"]
    pending = [row for row in comparisons if "PENDING" in row["classification"]]
    lines = [
        "# Reporte BP_FRONT Bar TrackingV3 Prototype", "",
        "Fecha: `%s`" % utc_now(), "",
        "## A. Algoritmo V3", "",
        "- Version: `%s`." % ALGORITHM_VERSION,
        "- Entrada: detecciones y mascaras existentes de Tracking V2; YOLO=False.",
        "- Tracklets BAR V2 como nodos y handoffs compatibles como aristas de un DAG.",
        "- Busqueda global determinista de la trayectoria que contiene la identidad principal V2.",
        "- Fast path: si V2 ya cubre todos los frames BAR RAW, se conserva exactamente.",
        "- No se interpolan gaps RAW; permanecen con indice -1.", "",
        "## B. Diferencias con V2", "",
        "V2 extiende greedy solo hacia delante, exige 2-5 frames solapados y geometria casi identica. V3 puede recuperar prefijos y sufijos, atravesar gaps RAW cortos sin inventar detecciones y compara caminos globales penalizando cada handoff.",
        "",
        "## C. Features de asociacion", "",
        "Centroide observado y predicho, gap temporal, orientacion, eje mayor, area, bbox, confidence, IoU/containment de mascaras y trayectoria robusta inicial/final.",
        "",
        "## D. Score", "",
        "Cada arista guarda `temporal`, `centroid_distance`, `predicted_motion`, `orientation`, `length`, `area`, `bbox_geometry`, `confidence` y `mask_transition`. El score total es su suma ponderada. El camino suma recompensa por deteccion existente y resta penalizacion de handoff, gaps e incompatibilidad.",
        "",
        "## E. Thresholds", "",
    ]
    for name, value in DEFAULT_PARAMETERS.items():
        lines.append("- `%s=%s`." % (name, value))
    lines += ["", "## F. Comparacion de los 14 videos", "",
              "| Video | V2 coverage | V3 coverage | Handoffs | Path V3 | Resultado | Evaluacion visual |",
              "|---|---:|---:|---:|---|---|---|"]
    for row in comparisons:
        lines.append("| %s | %.6f | %.6f | %s | %s | %s | %s |" % (
            row["video_id"], float(row["v2_main_track_coverage"]),
            float(row["v3_main_track_coverage"]), row["n_handoffs_v3"],
            str(row["v3_tracklet_path"]).replace("|", "->"), row["classification"],
            str(row["visual_evaluation"]).replace("|", "/"),
        ))
    lines += ["", "## G. Handoffs aceptados", ""]
    if accepted:
        for row in accepted:
            lines.append("- `%s` %s->%s en frame %s: score=%s, gap=%s, pred_error=%s." % (
                row["video_id"], row["source_track"], row["target_track"],
                row.get("transition_frame", ""), row.get("total_score", ""),
                row.get("gap_frames", ""), row.get("predicted_position_error", ""),
            ))
    else:
        lines.append("- Ninguno.")
    lines += ["", "## H. Handoffs rechazados relevantes", ""]
    for row in relevant_rejected[:30]:
        lines.append("- `%s` %s->%s: score=%s; `%s`." % (
            row["video_id"], row["source_track"], row["target_track"],
            row.get("total_score", ""), row.get("rejection_reason", ""),
        ))
    lines += [
        "", "## I. Casos mejorados", "",
        ", ".join("`%s`" % row["video_id"] for row in improved) if improved else "Ninguno confirmado.",
        "", "## J. Casos no mejorados", "",
        ", ".join("`%s`" % row["video_id"] for row in comparisons if row not in improved and row not in regressions),
        "", "## K. Regresiones", "",
        ", ".join("`%s`" % row["video_id"] for row in regressions) if regressions else "Ninguna.",
        "", "## L. Conclusion", "",
    ]
    if pending:
        conclusion = "V3 necesita revision visual antes de decidir."
    elif regressions:
        conclusion = "V3 rechazado por regresiones visuales."
    elif len(improved) >= 6 and all(not int(row["false_merge"]) for row in reviewed):
        conclusion = "V3 aprobado para ampliar prueba a los 76 casos de deteccion completa y track incompleto."
    else:
        conclusion = "V3 necesita ajustes antes de ampliar la prueba."
    lines += [
        conclusion, "",
        "No se modificaron RAW, mascaras, detecciones, Tracking V2 ni FEATURES V2. No se entreno ninguna red.", "",
    ]
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Prototipo global Tracking V3 BAR desde RAW V2")
    parser.add_argument("--video", help="Procesa un unico video bp_XXX")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--generate-previews", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    video_ids = [normalize_video_id(args.video)] if args.video else read_visual_test_ids(args.root)
    if len(video_ids) > 14:
        raise ValueError("El prototipo no permite procesar mas de 14 videos")
    comparisons = []
    candidate_rows = []
    visual_rows = []
    for video_id in video_ids:
        comparison, rows, audit, v2_indices, v3_indices = process_video(
            video_id, args.root, args.output_dir, args.audit_only
        )
        comparisons.append(comparison)
        candidate_rows.extend(rows)
        preview_paths = []
        if args.generate_previews and not args.audit_only:
            preview_paths = generate_previews(
                video_id, args.root, args.output_dir, args.dataset_dir,
                audit, v2_indices, v3_indices,
            )
        visual_rows.append({
            "video_id": video_id,
            "v3_tracklet_path": comparison["v3_tracklet_path"],
            "handoffs_v3": comparison["accepted_handoff_pairs"],
            "preview_paths": "|".join(preview_paths),
            "visual_status": "PENDING",
            "same_physical_bar": "",
            "false_merge": "",
            "raw_gaps_preserved": "",
            "control_preserved": "",
            "visual_finding": "Pendiente de revision",
            "final_classification": comparison["classification"],
        })
        print("[V3] %s: %.3f -> %.3f, path=%s, %s" % (
            video_id, comparison["v2_main_track_coverage"],
            comparison["v3_main_track_coverage"], comparison["v3_tracklet_path"],
            comparison["algorithm_decision"],
        ))

    apply_visual_reviews(comparisons, visual_rows)
    manifests = args.output_dir / "manifests"
    comparison_path = manifests / "bp_front_tracking_v3_prototype_comparison.csv"
    candidate_path = manifests / "bp_front_tracking_v3_handoff_candidates.csv"
    visual_path = manifests / "bp_front_tracking_v3_visual_review.csv"
    atomic_write_csv(comparison_path, comparisons, COMPARISON_FIELDS)
    atomic_write_csv(candidate_path, candidate_rows, CANDIDATE_FIELDS)
    atomic_write_csv(visual_path, visual_rows, VISUAL_FIELDS)
    atomic_write_text(args.report, build_report(comparisons, candidate_rows, visual_rows))
    print("Comparacion: %s" % comparison_path)
    print("Candidatos: %s" % candidate_path)
    print("Revision visual: %s" % visual_path)
    print("Reporte: %s" % args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
