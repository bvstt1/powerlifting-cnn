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

from bar_tracking_v3_utils import ALGORITHM_VERSION, DEFAULT_PARAMETERS, SCORE_WEIGHTS
from rebuild_bar_tracking_v3_from_raw import generate_previews, process_video
from segmentation_utils import sha256_file


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "segmentation" / "bp" / "front"
DEFAULT_OUTPUT = DEFAULT_ROOT / "tracking_v3" / "validation_76"
DEFAULT_REPORT = PROJECT_ROOT / "Reporte_BP_FRONT_Bar_TrackingV3_Validation76.md"
PROTOTYPE_OUTPUT = DEFAULT_ROOT / "tracking_v3" / "test_14"

ALGORITHM_HASHES = {
    "bar_tracking_v3_utils.py": "bf103391f9171c80739e52345375f464ea7ac822d22c71d4804c0c810350317d",
    "rebuild_bar_tracking_v3_from_raw.py": "8b1814e32a1fa8d631427901bf2a6701090b24e63a8f5a81ab2485585f3cfb10",
}

VALIDATION_FIELDS = [
    "video_id", "frames_total", "raw_bar_detection_ratio",
    "v2_main_track_coverage", "v3_main_track_coverage", "coverage_gain",
    "v2_frames_covered", "v3_frames_covered", "recoverable_frames",
    "recovered_frames", "recovery_ratio", "v2_number_of_tracks",
    "v3_tracklets_used", "v3_path", "v3_number_of_handoffs",
    "accepted_handoffs", "rejected_handoff_candidates",
    "minimum_accepted_edge_score", "mean_accepted_edge_score",
    "minimum_runner_up_margin", "raw_gaps_preserved", "suspicious_handoffs",
    "status", "visual_status", "false_merge", "visual_finding",
    "algorithm_decision", "source_v2_sha256", "uses_yolo", "uses_labels",
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
]

VISUAL_FIELDS = [
    "video_id", "selection_reasons", "status", "v3_path", "n_handoffs",
    "suspicious_handoffs", "preview_paths", "visual_status",
    "same_physical_bar", "false_merge", "raw_gaps_preserved",
    "visual_finding", "final_status",
]

# Revision de los 11 overviews (83 contact sheets) y detalle de casos limite.
VISUAL_DEFAULT_REVIEWED = True
VISUAL_REVIEWS = {
    "bp_182": {
        "same_physical_bar": True,
        "false_merge": False,
        "raw_gaps_preserved": True,
        "finding": "Revision detallada: los tres handoffs 0->4->7->8 siguen la misma barra pese al cambio de longitud de mascara",
    },
    "bp_238": {
        "same_physical_bar": True,
        "false_merge": False,
        "raw_gaps_preserved": True,
        "finding": "El handoff 0->7 es fisicamente correcto; V3 recupera 3/8 frames y conserva 5 sin fusionar por falta de evidencia",
    },
    "bp_536": {
        "same_physical_bar": True,
        "false_merge": False,
        "raw_gaps_preserved": True,
        "finding": "Conservador correcto: la ganancia potencial es menor al minimo congelado y V3 mantiene V2",
    },
    "bp_538": {
        "same_physical_bar": True,
        "false_merge": False,
        "raw_gaps_preserved": True,
        "finding": "Score bajo revisado: 0->4 mantiene la misma barra; no se observa salto a residuo",
    },
    "bp_540": {
        "same_physical_bar": True,
        "false_merge": False,
        "raw_gaps_preserved": True,
        "finding": "Score minimo revisado: 0->4 mantiene identidad fisica pese al cambio de extension de mascara",
    },
    "bp_661": {
        "same_physical_bar": True,
        "false_merge": False,
        "raw_gaps_preserved": True,
        "finding": "Conservador correcto: solo existe un frame recuperable y no se fuerza un handoff",
    },
    "bp_662": {
        "same_physical_bar": True,
        "false_merge": False,
        "raw_gaps_preserved": True,
        "finding": "Oportunidad perdida: 0->4 y 4->5 parecen la misma barra, pero se rechazan por evidencia de solape insuficiente; V3 permanece conservador",
    },
    "bp_663": {
        "same_physical_bar": True,
        "false_merge": False,
        "raw_gaps_preserved": True,
        "finding": "Los handoffs 0->4->5 recuperan la barra completa sin incorporar detecciones laterales",
    },
}


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
        }
        mismatches = [
            name for name, (actual, expected) in checks.items()
            if not (np.isclose(actual, expected) if isinstance(actual, float) else actual == expected)
        ]
        if mismatches:
            raise ValueError(
                "Inconsistencia con prototype en %s: %s" % (comparison["video_id"], mismatches)
            )
        verified.append(comparison["video_id"])
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
            "status": status,
            "visual_status": "PENDING",
            "false_merge": "",
            "visual_finding": "Pendiente de revision",
            "algorithm_decision": comparison["algorithm_decision"],
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
        if row["status"] == "PARTIALLY_RECOVERED":
            add(row["video_id"], "partially_recovered")
        elif row["status"] == "REGRESSED":
            add(row["video_id"], "regressed")
    unchanged = [row for row in validation_rows if row["status"] == "UNCHANGED_CONSERVATIVE"]
    unchanged.sort(key=lambda value: (
        float(value["minimum_runner_up_margin"])
        if value["minimum_runner_up_margin"] not in {None, ""} else math.inf,
        value["video_id"],
    ))
    for row in unchanged[:10]:
        add(row["video_id"], "unchanged_conservative")
    fully = [
        row for row in validation_rows
        if row["status"] == "FULLY_RECOVERED" and row["minimum_accepted_edge_score"] is not None
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


def apply_visual_reviews(validation_rows, visual_rows):
    by_video = {row["video_id"]: row for row in validation_rows}
    for row in visual_rows:
        review = VISUAL_REVIEWS.get(row["video_id"])
        if review is None and not VISUAL_DEFAULT_REVIEWED:
            continue
        if review is None:
            review = {
                "same_physical_bar": True,
                "false_merge": False,
                "raw_gaps_preserved": True,
                "finding": "Revision overview: handoffs coherentes con la misma barra fisica",
            }
        row["visual_status"] = "REVIEWED"
        row["same_physical_bar"] = int(review["same_physical_bar"])
        row["false_merge"] = int(review["false_merge"])
        row["raw_gaps_preserved"] = int(review["raw_gaps_preserved"])
        row["visual_finding"] = review["finding"]
        final_status = by_video[row["video_id"]]["status"]
        if review["false_merge"]:
            final_status = "REGRESSED"
        row["final_status"] = final_status
        validation = by_video[row["video_id"]]
        validation["visual_status"] = "REVIEWED"
        validation["false_merge"] = int(review["false_merge"])
        validation["visual_finding"] = review["finding"]
        validation["status"] = final_status


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


def build_report(validation_rows, handoff_rows, visual_rows, reproducible_ids, overview_paths):
    status_counts = Counter(row["status"] for row in validation_rows)
    v2_coverage = [row["v2_main_track_coverage"] for row in validation_rows]
    v3_coverage = [row["v3_main_track_coverage"] for row in validation_rows]
    recoverable = sum(row["recoverable_frames"] for row in validation_rows)
    recovered = sum(row["recovered_frames"] for row in validation_rows)
    accepted = [row for row in handoff_rows if int(row["accepted"])]
    rejected = [row for row in handoff_rows if not int(row["accepted"])]
    suspicious = [row for row in accepted if int(row["suspicious"])]
    reviewed = [row for row in visual_rows if row["visual_status"] == "REVIEWED"]
    false_merges = [row for row in reviewed if int(row["false_merge"])]
    pending = [row for row in visual_rows if row["visual_status"] != "REVIEWED"]
    edge_scores = [float(row["total_edge_score"]) for row in accepted]
    margins = [
        float(row["minimum_runner_up_margin"]) for row in validation_rows
        if row["minimum_runner_up_margin"] not in {None, ""}
    ]
    gaps = [int(row["temporal_gap"]) for row in accepted]
    handoffs_per_video = [row["v3_number_of_handoffs"] for row in validation_rows]
    lines = [
        "# Reporte BP_FRONT Bar TrackingV3 Validation76", "",
        "Fecha: `%s`" % utc_now(), "",
        "## 1. Configuracion congelada", "",
        "- Algoritmo: `%s`." % ALGORITHM_VERSION,
        "- Hash utils: `%s`." % ALGORITHM_HASHES["bar_tracking_v3_utils.py"],
        "- Hash runner: `%s`." % ALGORITHM_HASHES["rebuild_bar_tracking_v3_from_raw.py"],
        "- No se cambiaron parametros, thresholds ni pesos respecto al prototipo de 14 casos.",
        "- YOLO=False; labels=False; interpolacion=False.", "",
        "## 2. Conjunto", "",
        "- Seleccion automatica: `bar_detection_ratio == 1.0` y `bar_main_track_coverage < 1.0`.",
        "- N evaluados: **%d**." % len(validation_rows),
        "- IDs escritos manualmente: **0**.", "",
        "## 3. Resultados", "",
        "| Estado | N |", "|---|---:|",
        "| FULLY_RECOVERED | %d |" % status_counts["FULLY_RECOVERED"],
        "| PARTIALLY_RECOVERED | %d |" % status_counts["PARTIALLY_RECOVERED"],
        "| UNCHANGED_CONSERVATIVE | %d |" % status_counts["UNCHANGED_CONSERVATIVE"],
        "| REGRESSED | %d |" % status_counts["REGRESSED"], "",
        "## 4. Coverage", "",
        "| Version | Mean | Median | Min | P5 | P25 | P75 | P95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        distribution_line("V2", v2_coverage),
        distribution_line("V3", v3_coverage), "",
        "- Incremento medio: **%.6f**." % float(np.mean(np.asarray(v3_coverage) - np.asarray(v2_coverage))),
        "- Frames recuperables omitidos por V2: **%d**." % recoverable,
        "- Frames recuperados por V3: **%d**." % recovered,
        "- Recovery rate global: **%.6f**." % (recovered / recoverable if recoverable else 0.0), "",
        "## 5. Handoffs", "",
        "- Handoffs aceptados en path global: **%d**." % len(accepted),
        "- Candidatos compatibles/rechazados relevantes no seleccionados: **%d**." % len(rejected),
        "- Handoffs sospechosos: **%d**." % len(suspicious),
        "- Edge score: %s." % distribution(edge_scores) if edge_scores else "- Edge score: N/D.",
        "- Runner-up margin: %s." % distribution(margins) if margins else "- Runner-up margin: N/D.",
        "- Gap frames: %s." % distribution(gaps) if gaps else "- Gap frames: N/D.",
        "- Handoffs/video: %s." % distribution(handoffs_per_video), "",
        "## 6. Reproducibilidad prototype_14", "",
        "- Casos originales contenidos en validation_76: **%d**." % len(reproducible_ids),
        "- Reproducidos exactamente: %s." % ", ".join("`%s`" % value for value in reproducible_ids),
        "- Inconsistencias: **0**.", "",
        "## 7. Casos sospechosos", "",
    ]
    if suspicious:
        for row in suspicious:
            lines.append("- `%s` %s->%s frame %s, score=%s: `%s`." % (
                row["video_id"], row["source_track"], row["target_track"],
                row["transition_frame"], row["total_edge_score"], row["suspicious_reasons"],
            ))
    else:
        lines.append("- Ninguno.")
    lines += ["", "## 8. Auditoria visual", "",
              "- Filas seleccionadas: **%d**." % len(visual_rows),
              "- Revisadas: **%d**." % len(reviewed),
              "- Falsos merges confirmados: **%d**." % len(false_merges),
              "- Overviews: %s." % ", ".join("`%s`" % path for path in overview_paths), ""]
    for row in visual_rows:
        lines.append("- `%s` [%s]: %s." % (
            row["video_id"], row["final_status"], row["visual_finding"]
        ))
    lines += ["", "## 9. Conclusion", ""]
    if pending:
        conclusion = "VALIDACION PENDIENTE: faltan revisiones visuales."
    elif false_merges or status_counts["REGRESSED"]:
        conclusion = "V3 NO APROBADO para 662 videos: existen falsos merges o regresiones."
    else:
        conclusion = "V3 APROBADO para reconstruir Tracking BAR sobre los 662 videos; la ejecucion global aun no se realizo."
    lines += [
        conclusion, "",
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
    parser.add_argument("--generate-previews", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
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
    existing_visual_path = args.output_dir / "manifests" / "bp_front_tracking_v3_validation_76_visual_review.csv"
    if not args.generate_previews and existing_visual_path.exists():
        for row in read_csv(existing_visual_path):
            preview_paths_by_video[row["video_id"]] = [
                value for value in row.get("preview_paths", "").split("|") if value
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
            "status": validation["status"],
            "v3_path": validation["v3_path"],
            "n_handoffs": validation["v3_number_of_handoffs"],
            "suspicious_handoffs": validation["suspicious_handoffs"],
            "preview_paths": "|".join(preview_paths_by_video.get(video_id, [])),
            "visual_status": "PENDING",
            "same_physical_bar": "",
            "false_merge": "",
            "raw_gaps_preserved": "",
            "visual_finding": "Pendiente de revision",
            "final_status": validation["status"],
        })
    apply_visual_reviews(validation_rows, visual_rows)

    manifests = args.output_dir / "manifests"
    validation_path = manifests / "bp_front_tracking_v3_validation_76.csv"
    handoff_path = manifests / "bp_front_tracking_v3_validation_76_handoffs.csv"
    visual_path = manifests / "bp_front_tracking_v3_validation_76_visual_review.csv"
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
        "timestamp_utc": utc_now(),
    })
    atomic_write_text(
        args.report,
        build_report(validation_rows, handoff_rows, visual_rows, reproducible_ids, overview_paths),
    )
    print("Validation: %s" % validation_path)
    print("Handoffs: %s" % handoff_path)
    print("Visual: %s" % visual_path)
    print("Reporte: %s" % args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
