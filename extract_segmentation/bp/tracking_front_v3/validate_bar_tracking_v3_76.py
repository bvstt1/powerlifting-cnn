"""Evalúa el algoritmo V3 de barra sobre 76 casos BP front seleccionados.

Compara la continuidad V2/V3, registra handoffs, revisión visual y hashes.
Puede generar artefactos de validación; la evaluación no aprueba V3 para CNN.
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bar_tracking_v3_utils import ALGORITHM_VERSION, DEFAULT_PARAMETERS, SCORE_WEIGHTS
from code_versions import HISTORICAL_ALGORITHM_HASHES, RELOCATED_ALGORITHM_HASHES
from rebuild_bar_tracking_v3_from_raw import generate_previews, process_video
from segmentation_utils import sha256_file


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "segmentation" / "bp" / "front"
DEFAULT_OUTPUT = DEFAULT_ROOT / "tracking_v3" / "validation_76"
DEFAULT_REPORT = PROJECT_ROOT / "Reporte_BP_FRONT_Bar_TrackingV3_Validation76.md"
PROTOTYPE_OUTPUT = DEFAULT_ROOT / "tracking_v3" / "test_14"
DEFAULT_REVIEWS = (
    DEFAULT_OUTPUT / "manifests" / "bp_front_tracking_v3_validation_76_explicit_reviews.json"
)

ALGORITHM_HASHES = RELOCATED_ALGORITHM_HASHES

VALIDATION_FIELDS = [
    "video_id", "frames_total", "raw_bar_detection_ratio",
    "v2_main_track_coverage", "v3_main_track_coverage", "coverage_gain",
    "v2_frames_covered", "v3_frames_covered", "recoverable_frames",
    "recovered_frames", "recovery_ratio", "v2_number_of_tracks",
    "v3_tracklets_used", "v3_path", "v3_number_of_handoffs",
    "accepted_handoffs", "rejected_handoff_candidates",
    "minimum_accepted_edge_score", "mean_accepted_edge_score",
    "minimum_runner_up_margin", "raw_gaps_preserved", "suspicious_handoffs",
    "algorithm_status", "visual_review_status", "visual_identity_consistent",
    "false_merge_detected", "missed_valid_merge", "review_notes", "review_evidence",
    "algorithm_decision", "score_v2", "score_v3", "runner_up_score",
    "source_v2_sha256", "uses_yolo", "uses_labels",
]

HANDOFF_FIELDS = [
    "video_id", "source_track", "target_track", "transition_frame",
    "temporal_gap", "overlap_frames", "centroid_distance",
    "predicted_motion_error", "orientation_difference", "length_ratio",
    "area_ratio", "bbox_compatibility", "mask_transition_metric",
    "confidence_component", "temporal_score", "centroid_distance_score",
    "predicted_motion_score", "orientation_score", "length_score", "area_score",
    "bbox_geometry_score", "mask_transition_score", "total_edge_score",
    "runner_up_score", "runner_up_margin", "accepted", "selected_by_v3",
    "reason", "suspicious", "suspicious_reasons",
    "raw_gap_without_detection", "tracking_gap_with_raw_detection",
    "visual_review_status", "visual_identity_consistent", "false_merge_detected",
    "review_notes", "review_evidence",
]

VISUAL_FIELDS = [
    "video_id", "algorithm_status", "v2_coverage", "v3_coverage", "v3_path",
    "n_handoffs", "suspicious_handoffs", "visual_review_status",
    "visual_identity_consistent", "false_merge_detected", "missed_valid_merge",
    "raw_gap_preserved", "review_notes", "review_evidence", "selection_reasons",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


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


def select_validation_rows(root):
    rows = read_csv(root / "manifests" / "bp_front_bar_tracking_audit.csv")
    selected = [
        row for row in rows
        if np.isclose(float(row["bar_detection_ratio"]), 1.0)
        and float(row["bar_main_track_coverage"]) < 1.0 - 1e-12
    ]
    selected.sort(key=lambda row: row["video_id"])
    if len(selected) != 76:
        raise ValueError("Se esperaban 76 casos; encontrados %d" % len(selected))
    return selected


def verify_frozen_algorithm():
    paths = {
        "bar_tracking_v3_utils.py": SCRIPT_DIR / "bar_tracking_v3_utils.py",
        "rebuild_bar_tracking_v3_from_raw.py": SCRIPT_DIR / "rebuild_bar_tracking_v3_from_raw.py",
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    if actual != ALGORITHM_HASHES:
        raise ValueError("La implementación V3 congelada cambió: %s" % actual)
    expected_parameters = {
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
    if DEFAULT_PARAMETERS != expected_parameters:
        raise ValueError("Los parámetros V3 no coinciden con la configuración congelada")
    if ALGORITHM_VERSION != "bp_front_bar_tracking_v3_prototype_1":
        raise ValueError("Versión V3 inesperada")
    return actual


def status_for(frames_total, v2_frames, v3_frames):
    if v3_frames < v2_frames:
        return "REGRESSED"
    if v3_frames == frames_total:
        return "FULLY_RECOVERED"
    if v3_frames > v2_frames:
        return "PARTIALLY_RECOVERED"
    return "UNCHANGED_CONSERVATIVE"


def suspicious_reasons(edge, comparison, competitive_count):
    reasons = []
    score = float(edge["total_score"])
    margin = comparison.get("score_margin")
    if score <= DEFAULT_PARAMETERS["minimum_edge_score"] + 0.08:
        reasons.append("edge_score_near_threshold")
    if margin is not None and float(margin) <= DEFAULT_PARAMETERS["minimum_runner_up_margin"] + 0.15:
        reasons.append("runner_up_margin_near_threshold")
    if int(edge["gap_frames"]) >= 5:
        reasons.append("large_temporal_gap")
    if int(edge["overlap_frames"]) > 0:
        reasons.append("temporal_overlap")
    if float(edge["predicted_position_error"]) >= 0.10:
        reasons.append("high_predicted_motion_error")
    if float(edge["orientation_delta_deg"]) >= 25.0:
        reasons.append("large_orientation_change")
    if float(edge["major_axis_length_ratio"]) >= 1.8:
        reasons.append("large_length_change")
    if float(edge["area_ratio"]) >= 2.25:
        reasons.append("large_area_change")
    if competitive_count:
        reasons.append("competitive_handoff_candidates")
    if int(comparison["n_handoffs_v3"]) >= 3:
        reasons.append("three_or_more_handoffs")
    return reasons


def verify_prototype_reproducibility(comparisons, prototype_output):
    prototype_rows = read_csv(
        prototype_output / "manifests" / "bp_front_tracking_v3_prototype_comparison.csv"
    )
    prototype = {row["video_id"]: row for row in prototype_rows}
    verified = []
    for comparison in comparisons:
        reference = prototype.get(comparison["video_id"])
        if reference is None:
            continue
        checks = {
            "v3_path": (comparison["v3_path"], reference["v3_tracklet_path"]),
            "v3_coverage": (
                float(comparison["v3_main_track_coverage"]),
                float(reference["v3_main_track_coverage"]),
            ),
            "accepted_handoffs": (
                comparison["accepted_handoffs"], reference["accepted_handoff_pairs"]
            ),
            "score_v2": (float(comparison["score_v2"]), float(reference["score_v2"])),
            "score_v3": (float(comparison["score_v3"]), float(reference["score_v3"])),
            "score_margin": (
                float(comparison["minimum_runner_up_margin"]), float(reference["score_margin"])
            ),
            "algorithm_decision": (
                comparison["algorithm_decision"], reference["algorithm_decision"]
            ),
            "algorithm_status": (
                comparison["algorithm_status"],
                "FULLY_RECOVERED" if float(reference["v3_main_track_coverage"]) == 1.0
                else "PARTIALLY_RECOVERED",
            ),
        }
        mismatches = [
            name for name, (actual, expected) in checks.items()
            if not (
                np.isclose(actual, expected, rtol=0.0, atol=1e-12)
                if isinstance(actual, float) else actual == expected
            )
        ]
        if mismatches:
            raise ValueError(
                "Inconsistencia con prototype en %s: %s" % (comparison["video_id"], mismatches)
            )
        verified.append(comparison["video_id"])
    if len(verified) != 5:
        raise ValueError("Se esperaban 5 casos compartidos con prototype; encontrados %d" % len(verified))
    return verified


def build_validation_rows(selected_rows, processed):
    audit_by_video = {row["video_id"]: row for row in selected_rows}
    validation_rows = []
    handoff_rows = []
    audits_by_video = {}
    for comparison, candidate_rows, audit, v2_indices, v3_indices in processed:
        video_id = comparison["video_id"]
        source = audit_by_video[video_id]
        frames_total = int(source["frames_total"])
        v2_frames = int(np.count_nonzero(v2_indices >= 0))
        v3_frames = int(np.count_nonzero(v3_indices >= 0))
        recoverable = frames_total - v2_frames
        recovered = v3_frames - v2_frames
        status = status_for(frames_total, v2_frames, v3_frames)
        selected_edges = [row for row in candidate_rows if int(row.get("selected_by_v3", 0))]
        relevant_rejected = [
            row for row in candidate_rows
            if not int(row.get("selected_by_v3", 0))
            and float(row.get("total_score") or 0.0) > 0.0
        ]
        accepted_scores = [float(row["total_score"]) for row in selected_edges]
        validation_row = {
            "video_id": video_id,
            "frames_total": frames_total,
            "raw_bar_detection_ratio": float(source["bar_detection_ratio"]),
            "v2_main_track_coverage": v2_frames / frames_total,
            "v3_main_track_coverage": v3_frames / frames_total,
            "coverage_gain": (v3_frames - v2_frames) / frames_total,
            "v2_frames_covered": v2_frames,
            "v3_frames_covered": v3_frames,
            "recoverable_frames": recoverable,
            "recovered_frames": recovered,
            "recovery_ratio": recovered / recoverable if recoverable else 0.0,
            "v2_number_of_tracks": int(source["bar_number_of_tracks"]),
            "v3_tracklets_used": int(comparison["v3_n_tracklets_stitched"]),
            "v3_path": comparison["v3_tracklet_path"],
            "v3_number_of_handoffs": int(comparison["n_handoffs_v3"]),
            "accepted_handoffs": comparison["accepted_handoff_pairs"],
            "rejected_handoff_candidates": len(relevant_rejected),
            "minimum_accepted_edge_score": min(accepted_scores) if accepted_scores else None,
            "mean_accepted_edge_score": float(np.mean(accepted_scores)) if accepted_scores else None,
            "minimum_runner_up_margin": comparison["score_margin"],
            "raw_gaps_preserved": int(len(audit["raw_detection_gap_frames"]) == 0),
            "suspicious_handoffs": 0,
            "algorithm_status": status,
            "visual_review_status": "PENDING",
            "visual_identity_consistent": "",
            "false_merge_detected": "",
            "missed_valid_merge": "",
            "review_notes": "Pendiente de revision explicita",
            "review_evidence": "",
            "algorithm_decision": comparison["algorithm_decision"],
            "score_v2": comparison["score_v2"],
            "score_v3": comparison["score_v3"],
            "runner_up_score": (
                audit["runner_up_score"]["total"] if audit.get("runner_up_score") else None
            ),
            "source_v2_sha256": comparison["source_v2_sha256"],
            "uses_yolo": 0,
            "uses_labels": 0,
        }
        audits_by_video[video_id] = {
            "comparison": comparison,
            "candidate_rows": candidate_rows,
            "audit": audit,
            "v2_indices": v2_indices,
            "v3_indices": v3_indices,
        }

        for edge in candidate_rows:
            selected_edge = int(edge.get("selected_by_v3", 0))
            compatible_not_selected = int(edge.get("accepted", 0)) and not selected_edge
            if not selected_edge and not compatible_not_selected and float(edge.get("total_score") or 0.0) <= 0:
                continue
            competitive = 0
            if selected_edge:
                competitive = sum(
                    1 for other in candidate_rows
                    if other is not edge and int(other.get("accepted", 0))
                    and not int(other.get("selected_by_v3", 0))
                    and (
                        int(other["source_track"]) == int(edge["source_track"])
                        or int(other["target_track"]) == int(edge["target_track"])
                    )
                    and float(other.get("total_score") or 0.0) >= float(edge["total_score"]) - 0.10
                )
            suspicious = suspicious_reasons(edge, comparison, competitive) if selected_edge else []
            if suspicious:
                validation_row["suspicious_handoffs"] += 1
            reason = (
                "selected_global_path" if selected_edge else
                "compatible_not_selected_global_path" if compatible_not_selected else
                edge.get("rejection_reason", "rejected")
            )
            handoff_rows.append({
                "video_id": video_id,
                "source_track": edge["source_track"],
                "target_track": edge["target_track"],
                "transition_frame": edge.get("transition_frame", ""),
                "temporal_gap": edge.get("gap_frames", ""),
                "overlap_frames": edge.get("overlap_frames", ""),
                "centroid_distance": edge.get("centroid_distance_over_diagonal", ""),
                "predicted_motion_error": edge.get("predicted_position_error", ""),
                "orientation_difference": edge.get("orientation_delta_deg", ""),
                "length_ratio": edge.get("major_axis_length_ratio", ""),
                "area_ratio": edge.get("area_ratio", ""),
                "bbox_compatibility": edge.get("bbox_geometry_score", ""),
                "mask_transition_metric": edge.get("mask_transition_score", ""),
                "confidence_component": edge.get("confidence_score", ""),
                "temporal_score": edge.get("temporal_score", ""),
                "centroid_distance_score": edge.get("centroid_distance_score", ""),
                "predicted_motion_score": edge.get("predicted_motion_score", ""),
                "orientation_score": edge.get("orientation_score", ""),
                "length_score": edge.get("length_score", ""),
                "area_score": edge.get("area_score", ""),
                "bbox_geometry_score": edge.get("bbox_geometry_score", ""),
                "mask_transition_score": edge.get("mask_transition_score", ""),
                "total_edge_score": edge.get("total_score", ""),
                "runner_up_score": (
                    audit["runner_up_score"]["total"] if audit.get("runner_up_score") else ""
                ),
                "runner_up_margin": comparison["score_margin"],
                "accepted": selected_edge,
                "selected_by_v3": selected_edge,
                "reason": reason,
                "suspicious": int(bool(suspicious)),
                "suspicious_reasons": "|".join(suspicious),
                "raw_gap_without_detection": edge.get("raw_gap_without_detection", ""),
                "tracking_gap_with_raw_detection": edge.get("tracking_gap_with_raw_detection", ""),
                "visual_review_status": "PENDING" if selected_edge else "NOT_APPLICABLE",
                "visual_identity_consistent": "",
                "false_merge_detected": "",
                "review_notes": "Pendiente de revision explicita" if selected_edge else "",
                "review_evidence": "",
            })
        validation_rows.append(validation_row)
    return validation_rows, handoff_rows, audits_by_video


def visual_selection(validation_rows, handoff_rows):
    selected = {}

    def add(video_id, reason):
        selected.setdefault(video_id, set()).add(reason)

    accepted = [row for row in handoff_rows if int(row["accepted"])]
    for row in accepted:
        add(row["video_id"], "all_accepted_handoffs")
        if int(row["suspicious"]):
            add(row["video_id"], "suspicious_handoff")
    for row in sorted(validation_rows, key=lambda value: (-value["recovered_frames"], value["video_id"]))[:5]:
        add(row["video_id"], "highest_coverage_gain")
    for row in sorted(validation_rows, key=lambda value: (-value["v3_number_of_handoffs"], value["video_id"]))[:5]:
        add(row["video_id"], "highest_handoff_count")
    for row in validation_rows:
        if row["algorithm_status"] == "PARTIALLY_RECOVERED":
            add(row["video_id"], "partially_recovered")
        elif row["algorithm_status"] == "REGRESSED":
            add(row["video_id"], "regressed")
    unchanged = [
        row for row in validation_rows
        if row["algorithm_status"] == "UNCHANGED_CONSERVATIVE"
    ]
    unchanged.sort(key=lambda value: (
        float(value["minimum_runner_up_margin"])
        if value["minimum_runner_up_margin"] not in {None, ""} else math.inf,
        value["video_id"],
    ))
    for row in unchanged[:10]:
        add(row["video_id"], "unchanged_conservative")
    fully = [
        row for row in validation_rows
        if row["algorithm_status"] == "FULLY_RECOVERED"
        and row["minimum_accepted_edge_score"] is not None
    ]
    fully.sort(key=lambda value: (-value["minimum_accepted_edge_score"], value["video_id"]))
    for row in fully[:5]:
        add(row["video_id"], "high_score_fully_recovered")
    return selected


def create_overviews(preview_paths, output_root):
    overview_root = output_root / "previews" / "_overviews"
    overview_root.mkdir(parents=True, exist_ok=True)
    tiles = []
    for path in preview_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise OSError("No se pudo leer preview %s" % path)
        tile = cv2.resize(image, (640, 270))
        cv2.putText(tile, path.parent.parent.name + "/" + path.parent.name,
                    (8, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    outputs = []
    for group_index in range(0, len(tiles), 8):
        group = tiles[group_index:group_index + 8]
        while len(group) < 8:
            group.append(np.zeros_like(group[0]))
        overview = np.vstack([np.hstack(group[:4]), np.hstack(group[4:8])])
        path = overview_root / ("handoffs_overview_%02d.jpg" % (group_index // 8 + 1))
        if not cv2.imwrite(str(path), overview):
            raise OSError("No se pudo guardar %s" % path)
        outputs.append(path)
    return outputs


def load_explicit_reviews(path):
    if not path.exists():
        return {"video_reviews": [], "handoff_reviews": []}
    reviews = json.loads(path.read_text(encoding="utf-8"))
    if reviews.get("auto_review") is not False:
        raise ValueError("El registro de revisiones debe declarar auto_review=false")
    return reviews


def handoff_key(row):
    return (
        row["video_id"], int(row["source_track"]), int(row["target_track"]),
        int(row["transition_frame"]),
    )


def same_number(actual, expected):
    return np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-12)


def apply_explicit_reviews(validation_rows, handoff_rows, visual_rows, reviews):
    validation_by_video = {row["video_id"]: row for row in validation_rows}
    visual_by_video = {row["video_id"]: row for row in visual_rows}
    video_reviews = {row["video_id"]: row for row in reviews.get("video_reviews", [])}
    if len(video_reviews) != len(reviews.get("video_reviews", [])):
        raise ValueError("Hay revisiones de video duplicadas")

    handoff_reviews = {
        handoff_key(row): row for row in reviews.get("handoff_reviews", [])
    }
    if len(handoff_reviews) != len(reviews.get("handoff_reviews", [])):
        raise ValueError("Hay revisiones de handoff duplicadas")

    for video_id, review in video_reviews.items():
        if video_id not in validation_by_video or video_id not in visual_by_video:
            raise ValueError("Revision de video fuera del conjunto: %s" % video_id)
        validation = validation_by_video[video_id]
        checks = {
            "algorithm_status": validation["algorithm_status"] == review["algorithm_status"],
            "v2_coverage": same_number(validation["v2_main_track_coverage"], review["v2_coverage"]),
            "v3_coverage": same_number(validation["v3_main_track_coverage"], review["v3_coverage"]),
            "v3_path": validation["v3_path"] == review["v3_path"],
            "n_handoffs": int(validation["v3_number_of_handoffs"]) == int(review["n_handoffs"]),
        }
        if not all(checks.values()):
            raise ValueError("Revision visual obsoleta en %s: %s" % (
                video_id, [name for name, valid in checks.items() if not valid]
            ))
        evidence = [value for value in review["review_evidence"].split("|") if value]
        if not evidence or any(not (PROJECT_ROOT / value).exists() for value in evidence):
            raise ValueError("Evidencia visual ausente en %s" % video_id)
        for target in [validation, visual_by_video[video_id]]:
            target.update({
                "visual_review_status": review["visual_review_status"],
                "visual_identity_consistent": int(review["visual_identity_consistent"]),
                "false_merge_detected": int(review["false_merge_detected"]),
                "missed_valid_merge": int(review["missed_valid_merge"]),
                "review_notes": review["review_notes"],
                "review_evidence": review["review_evidence"],
            })
        visual_by_video[video_id]["raw_gap_preserved"] = int(review["raw_gap_preserved"])

    accepted_keys = set()
    for row in handoff_rows:
        if not int(row["accepted"]):
            continue
        key = handoff_key(row)
        accepted_keys.add(key)
        review = handoff_reviews.get(key)
        if review is None:
            continue
        checks = {
            "handoff_score": same_number(row["total_edge_score"], review["handoff_score"]),
            "suspicious_flag": int(row["suspicious"]) == int(review["suspicious_flag"]),
            "suspicious_reason": row["suspicious_reasons"] == review["suspicious_reason"],
        }
        if not all(checks.values()):
            raise ValueError("Revision de handoff obsoleta en %s: %s" % (
                key, [name for name, valid in checks.items() if not valid]
            ))
        evidence = PROJECT_ROOT / review["review_evidence"]
        if not evidence.exists():
            raise ValueError("Evidencia de handoff ausente: %s" % evidence)
        row.update({
            "visual_review_status": review["visual_review_status"],
            "visual_identity_consistent": int(review["visual_identity_consistent"]),
            "false_merge_detected": int(review["false_merge_detected"]),
            "review_notes": review["review_notes"],
            "review_evidence": review["review_evidence"],
        })

    extra_handoff_reviews = set(handoff_reviews) - accepted_keys
    if extra_handoff_reviews:
        raise ValueError("Revisiones de handoff fuera del path: %s" % sorted(extra_handoff_reviews))


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "min": float(values.min()), "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)), "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
    }


def distribution_line(name, values):
    stats = distribution(values)
    return "| %s | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f |" % (
        name, stats["mean"], stats["median"], stats["min"], stats["p5"],
        stats["p25"], stats["p75"], stats["p95"],
    )


def build_report(
    validation_rows, handoff_rows, visual_rows, reproducible_ids, overview_paths,
    verification=None,
):
    status_counts = Counter(row["algorithm_status"] for row in validation_rows)
    v2_coverage = [float(row["v2_main_track_coverage"]) for row in validation_rows]
    v3_coverage = [float(row["v3_main_track_coverage"]) for row in validation_rows]
    recoverable = sum(int(row["recoverable_frames"]) for row in validation_rows)
    recovered = sum(int(row["recovered_frames"]) for row in validation_rows)
    accepted = [row for row in handoff_rows if int(row["accepted"])]
    rejected = [row for row in handoff_rows if not int(row["accepted"])]
    suspicious = [row for row in accepted if int(row["suspicious"])]
    reviewed = [row for row in visual_rows if row["visual_review_status"] != "PENDING"]
    reviewed_handoffs = [
        row for row in accepted if row["visual_review_status"] != "PENDING"
    ]
    false_merge_handoffs = [
        row for row in reviewed_handoffs if int(row["false_merge_detected"])
    ]
    false_merge_videos = [
        row for row in reviewed if int(row["false_merge_detected"])
    ]
    pending = [row for row in visual_rows if row["visual_review_status"] == "PENDING"]
    pending_handoffs = [
        row for row in accepted if row["visual_review_status"] == "PENDING"
    ]
    evidence = [row["review_evidence"] for row in reviewed_handoffs]
    edge_scores = [float(row["total_edge_score"]) for row in accepted]
    margins = [
        float(row["minimum_runner_up_margin"]) for row in validation_rows
        if row["minimum_runner_up_margin"] not in {None, ""}
    ]
    gaps = [int(row["temporal_gap"]) for row in accepted]
    handoffs_per_video = [int(row["v3_number_of_handoffs"]) for row in validation_rows]
    lines = [
        "# Reporte BP_FRONT Bar TrackingV3 Validation76", "",
        "Fecha: `%s`" % utc_now(), "",
        "## Objetivo", "",
        "Cerrar formalmente Validation76 del algoritmo congelado mediante reejecucion determinista, revision visual explicita 76/76 y evidencia 1:1 para los 79 handoffs aceptados.", "",
        "## 1. Configuracion congelada", "",
        "- Algoritmo: `%s`." % ALGORITHM_VERSION,
        "- Hash utils (código reorganizado): `%s`." % ALGORITHM_HASHES["bar_tracking_v3_utils.py"],
        "- Hash runner (código reorganizado): `%s`." % ALGORITHM_HASHES["rebuild_bar_tracking_v3_from_raw.py"],
        "- Hash utils histórico: `%s`." % HISTORICAL_ALGORITHM_HASHES["bar_tracking_v3_utils.py"],
        "- Hash runner histórico: `%s`." % HISTORICAL_ALGORITHM_HASHES["rebuild_bar_tracking_v3_from_raw.py"],
        "- Los hashes históricos no equivalen byte a byte al código reorganizado.",
        "- No se cambiaron parametros, thresholds ni pesos respecto al prototipo de 14 casos.",
        "- YOLO=False; labels=False; interpolacion=False.", "",
        "### Parametros", "",
    ]
    for name, value in DEFAULT_PARAMETERS.items():
        lines.append("- `%s=%s`." % (name, value))
    lines += [
        "", "## 2. Correccion de auditoria", "",
        "- Se elimino el auto-review; ninguna fila sale de `PENDING` sin registro explicito.",
        "- `algorithm_status` permanece separado de `visual_review_status`.",
        "- Los resultados visuales no reescriben coverage, path, score ni decision V3.", "",
        "## 3. Conjunto", "",
        "- Seleccion automatica: `bar_detection_ratio == 1.0` y `bar_main_track_coverage < 1.0`.",
        "- N evaluados: **%d**." % len(validation_rows),
        "- IDs escritos manualmente: **0**.", "",
        "## 4. Resultados tecnicos", "",
        "| Estado | N |", "|---|---:|",
        "| FULLY_RECOVERED | %d |" % status_counts["FULLY_RECOVERED"],
        "| PARTIALLY_RECOVERED | %d |" % status_counts["PARTIALLY_RECOVERED"],
        "| UNCHANGED_CONSERVATIVE | %d |" % status_counts["UNCHANGED_CONSERVATIVE"],
        "| REGRESSED | %d |" % status_counts["REGRESSED"], "",
        "## 5. Coverage", "",
        "| Version | Mean | Median | Min | P5 | P25 | P75 | P95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        distribution_line("V2", v2_coverage),
        distribution_line("V3", v3_coverage), "",
        "- Incremento medio: **%.6f**." % float(np.mean(np.asarray(v3_coverage) - np.asarray(v2_coverage))),
        "- Frames recuperables omitidos por V2: **%d**." % recoverable,
        "- Frames recuperados por V3: **%d**." % recovered,
        "- Recovery rate global: **%.6f**." % (recovered / recoverable if recoverable else 0.0), "",
        "## 6. Handoffs", "",
        "- Handoffs aceptados en path global: **%d**." % len(accepted),
        "- Candidatos compatibles/rechazados relevantes no seleccionados: **%d**." % len(rejected),
        "- Handoffs sospechosos: **%d**." % len(suspicious),
        "- Handoffs revisados individualmente: **%d/%d**." % (len(reviewed_handoffs), len(accepted)),
        "- Evidencias unicas de handoff: **%d**." % len(set(evidence)),
        "- Falsos merges confirmados: **%d**." % len(false_merge_handoffs),
        "- Edge score: %s." % distribution(edge_scores) if edge_scores else "- Edge score: N/D.",
        "- Runner-up margin: %s." % distribution(margins) if margins else "- Runner-up margin: N/D.",
        "- Gap frames: %s." % distribution(gaps) if gaps else "- Gap frames: N/D.",
        "- Handoffs/video: %s." % distribution(handoffs_per_video), "",
        "## 7. Reproducibilidad prototype_14", "",
        "- Casos originales contenidos en validation_76: **%d**." % len(reproducible_ids),
        "- Reproducidos exactamente: %s." % ", ".join("`%s`" % value for value in reproducible_ids),
        "- Inconsistencias: **0**.", "",
        "## 8. Casos sospechosos", "",
    ]
    if suspicious:
        for row in suspicious:
            lines.append("- `%s` %s->%s frame %s, score=%s, review=%s: `%s`." % (
                row["video_id"], row["source_track"], row["target_track"],
                row["transition_frame"], row["total_edge_score"],
                row["visual_review_status"], row["suspicious_reasons"],
            ))
    else:
        lines.append("- Ninguno.")
    lines += ["", "## 9. Auditoria visual", "",
              "- Videos revisados explicitamente: **%d/%d**." % (len(reviewed), len(visual_rows)),
              "- Videos PENDING: **%d**." % len(pending),
              "- Handoffs PENDING: **%d**." % len(pending_handoffs),
              "- Videos con falso merge: **%d**." % len(false_merge_videos),
              "- Handoffs con falso merge: **%d**." % len(false_merge_handoffs),
              "- RAW gaps preservados: **%d/%d** (el filtro contiene deteccion RAW completa)." % (
                  sum(int(row["raw_gap_preserved"]) for row in visual_rows), len(visual_rows)
              ),
              "- Los overviews son auxiliares y no cuentan como evidencia 1:1.", ""]
    for row in visual_rows:
        lines.append("- `%s` [%s / %s]: %s Evidencia: `%s`." % (
            row["video_id"], row["algorithm_status"], row["visual_review_status"],
            row["review_notes"], row["review_evidence"],
        ))

    lines += ["", "## 10. Casos prioritarios", ""]
    priority = {"bp_182", "bp_238", "bp_536", "bp_538", "bp_540", "bp_544", "bp_661", "bp_662", "bp_663"}
    for row in visual_rows:
        if row["video_id"] in priority:
            lines.append("- `%s`: %s" % (row["video_id"], row["review_notes"]))

    lines += ["", "## 11. Integridad y verificador", ""]
    if verification is None:
        lines.append("- Verificador read-only: **PENDIENTE**.")
    else:
        integrity = verification.get("integrity", {})
        lines += [
            "- Verificador read-only: **%s**." % verification.get("result", "UNKNOWN"),
            "- Checks aprobados: **%s/%s**." % (
                verification.get("checks_passed", 0), verification.get("checks_total", 0)
            ),
            "- Integridad PRE/POST: **%s**." % verification.get("integrity_pre_post", "UNKNOWN"),
            "- Prototype suite: **%s**." % verification.get("prototype_suite", "UNKNOWN"),
            "- Artefactos RAW/V2/features/audits comparados PRE/POST: **%s**; discrepancias: **%s**." % (
                integrity.get("source_artifacts_checked", "N/D"),
                len(integrity.get("source_mismatches", [])),
            ),
            "- Hash utils PRE: `%s`; POST: `%s`." % (
                integrity.get("algorithm_hashes_pre", {}).get("bar_tracking_v3_utils.py", "N/D"),
                integrity.get("algorithm_hashes_post", {}).get("bar_tracking_v3_utils.py", "N/D"),
            ),
            "- Hash runner PRE: `%s`; POST: `%s`." % (
                integrity.get("algorithm_hashes_pre", {}).get("rebuild_bar_tracking_v3_from_raw.py", "N/D"),
                integrity.get("algorithm_hashes_post", {}).get("rebuild_bar_tracking_v3_from_raw.py", "N/D"),
            ),
            "- Hash auditor PRE: `%s`; POST: `%s` (cambio permitido y limitado a auditoria)." % (
                integrity.get("validator_hash_pre", "N/D"),
                integrity.get("validator_hash_post", "N/D"),
            ),
        ]

    lines += ["", "## 12. Conclusion", ""]
    if pending or pending_handoffs:
        conclusion = "VALIDATION76_NOT_APPROVED: faltan revisiones visuales explicitas."
    elif false_merge_handoffs or status_counts["REGRESSED"]:
        conclusion = (
            "VALIDATION76_NOT_APPROVED: se confirmaron %d falsos merges visuales en %d video(s)."
            % (len(false_merge_handoffs), len(false_merge_videos))
        )
    elif verification is not None and verification.get("result") != "OK":
        conclusion = "VALIDATION76_NOT_APPROVED: el verificador final no termino en OK."
    else:
        conclusion = "VALIDATION76_APPROVED."
    lines += [
        "**%s**" % conclusion, "",
        "No se modificaron RAW, mascaras, detecciones, Tracking V2 ni FEATURES V2.",
        "No se procesaron los 662 videos y no se generaron FEATURES V3 globales.", "",
    ]
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Valida Tracking V3 congelado sobre los 76 casos fragmentados")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "dataset" / "bp" / "front")
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--generate-previews", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifests = args.output_dir / "manifests"
    validation_path = manifests / "bp_front_tracking_v3_validation_76.csv"
    handoff_path = manifests / "bp_front_tracking_v3_validation_76_handoffs.csv"
    visual_path = manifests / "bp_front_tracking_v3_validation_76_visual_review.csv"
    verification_path = manifests / "bp_front_tracking_v3_validation_76_verification.json"
    if args.report_only:
        validation_rows = read_csv(validation_path)
        handoff_rows = read_csv(handoff_path)
        visual_rows = read_csv(visual_path)
        prototype_ids = {
            row["video_id"] for row in read_csv(
                PROTOTYPE_OUTPUT / "manifests" / "bp_front_tracking_v3_prototype_comparison.csv"
            )
        }
        reproducible_ids = sorted(prototype_ids & {row["video_id"] for row in validation_rows})
        overview_paths = sorted((args.output_dir / "previews" / "_overviews").glob("*.jpg"))
        verification = (
            json.loads(verification_path.read_text(encoding="utf-8"))
            if verification_path.exists() else None
        )
        atomic_write_text(
            args.report,
            build_report(
                validation_rows, handoff_rows, visual_rows, reproducible_ids,
                overview_paths, verification,
            ),
        )
        print("Reporte: %s" % args.report)
        return 0

    hashes = verify_frozen_algorithm()
    selected_rows = select_validation_rows(args.root)
    processed = []
    for position, source in enumerate(selected_rows, 1):
        video_id = source["video_id"]
        result = process_video(video_id, args.root, args.output_dir, audit_only=False)
        processed.append(result)
        print("[%d/76] %s: %.3f -> %.3f" % (
            position, video_id,
            result[0]["v2_main_track_coverage"], result[0]["v3_main_track_coverage"],
        ))

    validation_rows, handoff_rows, audits_by_video = build_validation_rows(selected_rows, processed)
    reproducible_ids = verify_prototype_reproducibility(validation_rows, PROTOTYPE_OUTPUT)
    selected_visual = visual_selection(validation_rows, handoff_rows)
    preview_paths_by_video = {}
    all_contact_paths = []
    if not args.generate_previews:
        for video_id in selected_visual:
            preview_paths_by_video[video_id] = [
                str(path.resolve()) for path in sorted(
                    (args.output_dir / "previews" / video_id).glob("*/*_contact_sheet.jpg")
                )
            ]
    if args.generate_previews:
        for video_id in sorted(selected_visual):
            values = audits_by_video[video_id]
            preview_audit = dict(values["audit"])
            if not preview_audit["accepted_handoffs"] and not preview_audit.get("proposed_handoffs"):
                rejected = sorted(
                    [
                        candidate for candidate in preview_audit["handoff_candidates"]
                        if candidate.get("transition_frame") is not None
                        and float(candidate.get("total_score", 0.0)) > 0.0
                    ],
                    key=lambda candidate: (
                        -float(candidate.get("total_score", 0.0)),
                        candidate["source_track"], candidate["target_track"],
                    ),
                )
                preview_audit["proposed_handoffs"] = rejected[:2]
            paths = generate_previews(
                video_id, args.root, args.output_dir, args.dataset_dir,
                preview_audit, values["v2_indices"], values["v3_indices"],
            )
            preview_paths_by_video[video_id] = paths
            all_contact_paths.extend(Path(path) for path in paths)
    if all_contact_paths:
        overview_paths = create_overviews(all_contact_paths, args.output_dir)
    else:
        overview_paths = sorted((args.output_dir / "previews" / "_overviews").glob("*.jpg"))

    validation_by_video = {row["video_id"]: row for row in validation_rows}
    visual_rows = []
    for video_id, reasons in sorted(selected_visual.items()):
        validation = validation_by_video[video_id]
        visual_rows.append({
            "video_id": video_id,
            "selection_reasons": "|".join(sorted(reasons)),
            "algorithm_status": validation["algorithm_status"],
            "v2_coverage": validation["v2_main_track_coverage"],
            "v3_coverage": validation["v3_main_track_coverage"],
            "v3_path": validation["v3_path"],
            "n_handoffs": validation["v3_number_of_handoffs"],
            "suspicious_handoffs": validation["suspicious_handoffs"],
            "visual_review_status": "PENDING",
            "visual_identity_consistent": "",
            "false_merge_detected": "",
            "missed_valid_merge": "",
            "raw_gap_preserved": "",
            "review_notes": "Pendiente de revision explicita",
            "review_evidence": "",
        })
    explicit_reviews = load_explicit_reviews(args.reviews)
    apply_explicit_reviews(validation_rows, handoff_rows, visual_rows, explicit_reviews)

    atomic_write_csv(validation_path, validation_rows, VALIDATION_FIELDS)
    atomic_write_csv(handoff_path, handoff_rows, HANDOFF_FIELDS)
    atomic_write_csv(visual_path, visual_rows, VISUAL_FIELDS)
    atomic_write_json(manifests / "bp_front_tracking_v3_validation_76_config.json", {
        "algorithm_version": ALGORITHM_VERSION,
        "algorithm_hashes": hashes,
        "parameters": DEFAULT_PARAMETERS,
        "score_weights": SCORE_WEIGHTS,
        "selection": "bar_detection_ratio == 1.0 and bar_main_track_coverage < 1.0",
        "n_selected": len(selected_rows),
        "uses_yolo": False,
        "uses_labels": False,
        "auto_review": False,
        "explicit_reviews_path": str(args.reviews.resolve()),
        "timestamp_utc": utc_now(),
    })
    atomic_write_text(
        args.report,
        build_report(
            validation_rows, handoff_rows, visual_rows, reproducible_ids, overview_paths,
            None,
        ),
    )
    print("Validation: %s" % validation_path)
    print("Handoffs: %s" % handoff_path)
    print("Visual: %s" % visual_path)
    print("Reporte: %s" % args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
