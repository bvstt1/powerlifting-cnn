"""Audita continuidad de la barra en el tracking frontal V2.2.

Compara detecciones RAW con la cobertura del track principal, reúne casos
para revisión visual y genera informe; no reconstruye tracks ni entrena CNN.
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DISPLAY_DIR = PROJECT_ROOT / "display" / "bp" / "fast"
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(DISPLAY_DIR))

from inspect_segmentation_front_bp import (  # noqa: E402
    annotate_frame,
    default_options,
    read_video_frame,
)


DEFAULT_ROOT = PROJECT_ROOT / "segmentation" / "bp" / "front"
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "bp" / "front"
DEFAULT_REPORT = PROJECT_ROOT / "Reporte_BP_FRONT_Bar_Tracking_Audit_V1.md"
PERCENTILES = [1, 5, 25, 50, 75, 95, 99]

AUDIT_FIELDS = [
    "video_id", "audit_category", "continuity_pattern", "diagnosis",
    "visual_review_recommended", "frames_total", "bar_detection_ratio",
    "bar_main_track_coverage", "bar_detection_minus_coverage",
    "bar_frames_detected", "bar_frames_missing", "bar_main_frames",
    "bar_main_missing_with_raw_detection", "bar_number_of_tracks",
    "bar_main_track_id", "bar_original_track_ids", "bar_track_intervals",
    "bar_longest_missing_gap", "bar_mean_confidence", "bar_median_confidence",
    "bar_min_confidence", "bar_max_confidence", "bar_multiple_instance_frames",
    "bar_max_instances", "bar_stitching_applied", "bar_accepted_handoffs",
    "bar_rejected_handoffs", "bar_main_temporal_mask_iou_min",
    "bar_main_temporal_bbox_iou_min", "bar_main_centroid_jump_max_normalized",
    "bar_main_area_ratio_max", "bar_warning_reasons", "source_raw_unchanged",
]

VISUAL_FIELDS = [
    "video_id", "selection_reasons", "preview_frames", "visual_classification",
    "visual_finding", "future_temporal_feature_risk",
]

# Revision manual de los contact sheets, basada solo en video, mascaras y tracks.
VISUAL_REVIEWS = {
    "bp_001": {
        "classification": "E_MAIN_TRACK_REPRESENTS_RELEVANT_TRAJECTORY",
        "finding": "Control correcto: una sola identidad BAR estable y mascara coherente durante el movimiento",
        "risk": "LOW",
    },
    "bp_004": {
        "classification": "B_TRACK_ID_CHANGE+C_PARTIAL_DUPLICATE+F_SEVERE",
        "finding": "La barra esta segmentada durante toda la secuencia, pero se reparte entre tracks 0, 3 y 4; el principal solo conserva frames 20-47",
        "risk": "HIGH",
    },
    "bp_019": {
        "classification": "B_TRACK_ID_CHANGE+F_SEVERE",
        "finding": "La barra fisica continua bajo track 4 al final; el handoff fue rechazado y FEATURES pierde 13 frames con deteccion disponible",
        "risk": "HIGH",
    },
    "bp_067": {
        "classification": "B_TRACK_ID_CHANGE+F_SEVERE",
        "finding": "Transicion clara de track 0 a 3 sobre la misma barra fisica; el principal omite los 22 frames finales aunque la mascara sigue siendo util",
        "risk": "HIGH",
    },
    "bp_139": {
        "classification": "A_REAL_RAW_LOSS+B_LATE_TRACK_FRAGMENT",
        "finding": "La barra sale parcialmente del encuadre al final y hay nueve frames RAW ausentes; reaparece una deteccion residual bajo otro track",
        "risk": "MEDIUM",
    },
    "bp_198": {
        "classification": "A_REAL_RAW_LOSS+B_TRACK_ID_CHANGE+F_SEVERE",
        "finding": "Hay dos frames de perdida RAW, pero la mayor diferencia proviene de tracks 5 y 6 que continúan la barra mientras el principal termina en frame 26",
        "risk": "HIGH",
    },
    "bp_241": {
        "classification": "A_REAL_RAW_LOSS+E_MAIN_MATCHES_RAW",
        "finding": "El principal coincide exactamente con la disponibilidad RAW; los nueve frames faltantes aparecen al salir la barra del encuadre",
        "risk": "MEDIUM",
    },
    "bp_301": {
        "classification": "A_SHORT_RAW_GAP+B_TRACK_ID_CHANGE+F_SEVERE",
        "finding": "Tras un gap corto por desenfoque, track 4 recupera correctamente la barra pero el principal permanece en track 0 y omite 22 frames detectados",
        "risk": "HIGH",
    },
    "bp_467": {
        "classification": "A_REAL_RAW_LOSS+B_TRACK_ID_CHANGE+F_SEVERE",
        "finding": "Un gap visible separa tracks 0 y 4; la barra vuelve a estar correctamente segmentada, pero esos 14 frames no entran al principal",
        "risk": "HIGH",
    },
    "bp_469": {
        "classification": "A_REAL_RAW_LOSS+B_LATE_TRACK_FRAGMENT",
        "finding": "La mayor parte de la trayectoria permanece en track 0; tras gaps finales aparecen tracks cortos 4 y 5 sobre la barra real",
        "risk": "MEDIUM",
    },
    "bp_518": {
        "classification": "C_PARTIAL_REPRESENTATIONS+D_RESIDUAL_FALSE_BAR+E_MAIN_CORRECT",
        "finding": "El principal track 0 cubre toda la barra; las instancias laterales son fragmentos o detecciones residuales y no desplazan la trayectoria principal",
        "risk": "LOW",
    },
    "bp_520": {
        "classification": "B_AUDITED_HANDOFF+C_PARTIAL_REPRESENTATIONS+E_MAIN_CORRECT",
        "finding": "El handoff original 0 a 5 corresponde a la misma barra y el track virtual 7 mantiene cobertura completa; los laterales residuales no alteran el principal",
        "risk": "LOW",
    },
    "bp_541": {
        "classification": "B_TRACK_ID_CHANGE+F_SEVERE",
        "finding": "Tracks 0 y 5 representan consecutivamente la misma barra, pero el principal final solo toma track 5 y pierde 87 frames iniciales disponibles",
        "risk": "HIGH",
    },
    "bp_544": {
        "classification": "B_TRACK_ID_CHANGE+D_RESIDUAL_FALSE_BAR+F_SEVERE",
        "finding": "La barra cambia de track 0 a 5 con continuidad visual; detecciones laterales residuales coexisten y el principal omite 96 frames disponibles",
        "risk": "HIGH",
    },
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_csv(path, rows, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_write_text(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def as_int(value):
    return int(float(value)) if value not in {None, ""} else 0


def as_float(value):
    return float(value) if value not in {None, ""} else math.nan


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0, "mean": None, "median": None, "std": None,
                "min": None, "max": None, "percentiles": {str(p): None for p in PERCENTILES}}
    return {
        "count": int(values.size), "mean": float(values.mean()),
        "median": float(np.median(values)), "std": float(values.std()),
        "min": float(values.min()), "max": float(values.max()),
        "percentiles": {str(p): float(np.percentile(values, p)) for p in PERCENTILES},
    }


def fmt(value, digits=6):
    if value is None or not np.isfinite(float(value)):
        return "N/D"
    return ("%%.%df" % digits) % float(value)


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


def take_high_with_ties(rows, metric, limit):
    ordered = sorted(rows, key=lambda row: (-float(row[metric]), row["video_id"]))
    if len(ordered) <= limit:
        return ordered
    cutoff = float(ordered[limit - 1][metric])
    return [row for row in ordered if float(row[metric]) >= cutoff - 1e-12]


def corrected_qa_selection(class_rows, features_root, audits_root):
    selected = defaultdict(set)
    for class_name in ["bar", "bench", "person"]:
        rows = [row for row in class_rows if row["class_name"] == class_name]
        for metric in ["detection_ratio", "main_track_coverage", "confidence_mean"]:
            for row in objectively_low_rows(rows, metric, 3):
                selected[row["video_id"]].add("low_%s_%s" % (class_name, metric))
        for row in take_high_with_ties(rows, "frames_multiple_instances", 3):
            if int(row["frames_multiple_instances"]) > 0:
                selected[row["video_id"]].add("multiple_%s" % class_name)
        for row in take_high_with_ties(rows, "longest_missing_gap", 3):
            if int(row["longest_missing_gap"]) > 0:
                selected[row["video_id"]].add("long_gap_%s" % class_name)

    stitching = []
    for audit_path in audits_root.glob("bp_*_tracking_audit.json"):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        merge = audit["merge_audit"]
        score = (
            len(merge["accepted_stitches"])
            + len(merge["bench_identity"]["fallback_frames"])
            + len(merge["bar_identity"]["accepted_handoffs"])
        )
        stitching.append({"video_id": audit["video_id"], "score": score})
    for row in take_high_with_ties(stitching, "score", 5):
        if row["score"]:
            selected[row["video_id"]].add("high_stitching")

    distance_candidates = defaultdict(list)
    for path in sorted(features_root.glob("bp_*_features.npz")):
        with np.load(path, allow_pickle=False) as data:
            names = data["interaction_feature_names"].astype(str).tolist()
            pairs = data["interaction_pair_names"].astype(str).tolist()
            distance_index = names.index("minimum_mask_distance_normalized")
            both_index = names.index("both_detected")
            for pair_index, pair_name in enumerate(pairs):
                values = data["interaction_features"][:, pair_index]
                available = values[:, both_index] == 1.0
                distances = values[available, distance_index]
                distances = distances[np.isfinite(distances)]
                if distances.size:
                    distance_candidates[pair_name].append({
                        "video_id": path.name[:6], "distance": float(distances.max()),
                    })
    for pair_name, rows in distance_candidates.items():
        for row in take_high_with_ties(rows, "distance", 3):
            selected[row["video_id"]].add("extreme_distance_%s" % pair_name)
    return selected


def update_qa_metadata(root, quality_rows):
    class_rows = []
    for row in quality_rows:
        if row["record_type"] != "class":
            continue
        converted = dict(row)
        for field in ["detection_ratio", "main_track_coverage", "confidence_mean"]:
            converted[field] = as_float(row[field])
        for field in ["frames_multiple_instances", "longest_missing_gap"]:
            converted[field] = as_int(row[field])
        class_rows.append(converted)
    selected = corrected_qa_selection(
        class_rows,
        root / "tracking_v2" / "features",
        root / "tracking_v2" / "audits",
    )
    rows = [
        {"video_id": video_id, "selection_reasons": "|".join(sorted(reasons))}
        for video_id, reasons in sorted(selected.items())
    ]
    path = root / "manifests" / "bp_front_qa_preview_selection.csv"
    atomic_write_csv(path, rows, ["video_id", "selection_reasons"])

    report_path = PROJECT_ROOT / "Reporte_BP_FRONT_Segmentation_Dataset_V1.md"
    report = report_path.read_text(encoding="utf-8")
    start = report.index("## 8. Previews QA")
    end = report.index("## 9. Errores")
    section = [
        "## 8. Previews QA", "",
        "Se seleccionaron automaticamente %d videos sin usar labels." % len(rows),
        "La seleccion exige un valor objetivamente no perfecto y conserva empates en el cutoff.",
    ]
    for row in rows:
        section.append("- `%s`: %s." % (
            row["video_id"], row["selection_reasons"].replace("|", ", ")
        ))
    updated = report[:start] + "\n".join(section) + "\n\n" + report[end:]
    atomic_write_text(report_path, updated)
    return rows


def update_bp349_manifest(root):
    path = root / "manifests" / "bp_front_segmentation_manifest.csv"
    rows = read_csv(path)
    found = False
    for row in rows:
        if row["video_id"] == "bp_349":
            if row["quality_status"] != "QUALITY_ERROR" or row["status"] != "ERROR":
                raise ValueError("bp_349 no conserva su estado de error")
            row["quality_reasons"] = "degenerate_polygon"
            found = True
    if not found:
        raise ValueError("bp_349 no existe en el manifest")
    atomic_write_csv(path, rows, list(rows[0]))


def intervals_text(tracks):
    return "|".join(
        "%s:%s-%s(%s)" % (
            track["track_id"], track["first_frame"], track["last_frame"], track["frames"]
        )
        for track in sorted(tracks, key=lambda item: (item["first_frame"], item["track_id"]))
    )


def classify_case(detection, coverage, longest_gap):
    if coverage < 0.60 or detection < 0.95 or longest_gap >= 10:
        return "E_extreme_visual_review"
    if np.isclose(detection, 1.0) and np.isclose(coverage, 1.0):
        return "A_complete_detection_complete_track"
    if np.isclose(detection, 1.0):
        return "B_complete_detection_fragmented_track"
    if detection - coverage <= 0.05:
        return "C_incomplete_detection_reasonable_track"
    return "D_incomplete_detection_fragmented_track"


def diagnose_case(detection, coverage, multiple_frames, accepted, rejected, tracks):
    if accepted:
        return "accepted_bar_handoff"
    if coverage + 1e-12 < detection:
        if rejected:
            return "fragmented_track_ids_rejected_handoff"
        if multiple_frames:
            return "fragmented_track_ids_with_competing_instances"
        return "fragmented_track_ids"
    if detection < 1.0 - 1e-12:
        return "raw_detection_gaps_only"
    if multiple_frames:
        return "continuous_main_with_residual_multiple_instances"
    if len(tracks) > 1:
        return "continuous_main_with_residual_tracks"
    return "continuous_single_track"


def build_audit_rows(root, quality_rows):
    bar_quality = {
        row["video_id"]: row for row in quality_rows
        if row["record_type"] == "class" and row["class_name"] == "bar"
    }
    rows = []
    for video_id in sorted(bar_quality):
        quality = bar_quality[video_id]
        audit_path = root / "tracking_v2" / "audits" / (video_id + "_tracking_audit.json")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        tracks = audit["tracks_after_by_class"]["bar"]
        main_id = int(audit["selected_tracks_after"]["bar"])
        main = next(track for track in tracks if int(track["track_id"]) == main_id)
        merge = audit["merge_audit"]["bar_identity"]
        accepted = merge["accepted_handoffs"]
        rejected = merge["rejected_handoffs"]
        detection = as_float(quality["detection_ratio"])
        coverage = as_float(quality["main_track_coverage"])
        frames_detected = as_int(quality["frames_detected"])
        main_frames = int(main["frames"])
        multiple = as_int(quality["frames_multiple_instances"])
        original_ids = main.get("original_track_ids", [])
        temporal_mask_min = main.get("temporal_mask_iou", {}).get("min")
        temporal_bbox_min = main.get("temporal_bbox_iou", {}).get("min")
        features_path = root / "tracking_v2" / "features" / (video_id + "_features.npz")
        with np.load(features_path, allow_pickle=False) as features:
            class_position = features["class_names"].astype(str).tolist().index("bar")
            names = features["object_feature_names"].astype(str).tolist()
            values = features["object_features"][:, class_position]
            dx = values[:, names.index("centroid_dx_normalized")]
            dy = values[:, names.index("centroid_dy_normalized")]
            finite_jump = np.isfinite(dx) & np.isfinite(dy)
            centroid_jump_max = (
                float(np.max(np.hypot(dx[finite_jump], dy[finite_jump])))
                if np.any(finite_jump) else None
            )
            areas = values[:, names.index("mask_area_px")]
            detected = np.isfinite(areas)
            consecutive = detected[1:] & detected[:-1] & (areas[1:] > 0) & (areas[:-1] > 0)
            area_ratio_max = (
                float(np.max(np.maximum(areas[1:][consecutive], areas[:-1][consecutive])
                             / np.minimum(areas[1:][consecutive], areas[:-1][consecutive])))
                if np.any(consecutive) else None
            )
        visual = (
            coverage < 0.80 or detection < 1.0 - 1e-12 or multiple > 0
            or bool(accepted) or bool(rejected)
        )
        rows.append({
            "video_id": video_id,
            "audit_category": classify_case(detection, coverage, as_int(quality["longest_missing_gap"])),
            "continuity_pattern": (
                "complete_detection_incomplete_main" if np.isclose(detection, 1.0) and coverage < 1.0 - 1e-12
                else "incomplete_detection_incomplete_main" if detection < 1.0 - 1e-12 and coverage < detection - 1e-12
                else "incomplete_detection_main_matches_raw" if detection < 1.0 - 1e-12
                else "complete_detection_complete_main"
            ),
            "diagnosis": diagnose_case(detection, coverage, multiple, accepted, rejected, tracks),
            "visual_review_recommended": int(visual),
            "frames_total": as_int(quality["frames_total"]),
            "bar_detection_ratio": detection,
            "bar_main_track_coverage": coverage,
            "bar_detection_minus_coverage": detection - coverage,
            "bar_frames_detected": frames_detected,
            "bar_frames_missing": as_int(quality["frames_missing"]),
            "bar_main_frames": main_frames,
            "bar_main_missing_with_raw_detection": frames_detected - main_frames,
            "bar_number_of_tracks": as_int(quality["number_of_tracks"]),
            "bar_main_track_id": main_id,
            "bar_original_track_ids": "|".join(map(str, original_ids)),
            "bar_track_intervals": intervals_text(tracks),
            "bar_longest_missing_gap": as_int(quality["longest_missing_gap"]),
            "bar_mean_confidence": as_float(quality["confidence_mean"]),
            "bar_median_confidence": as_float(quality["confidence_median"]),
            "bar_min_confidence": as_float(quality["confidence_min"]),
            "bar_max_confidence": as_float(quality["confidence_max"]),
            "bar_multiple_instance_frames": multiple,
            "bar_max_instances": as_int(quality["max_instances"]),
            "bar_stitching_applied": as_int(quality["stitching_applied"]),
            "bar_accepted_handoffs": len(accepted),
            "bar_rejected_handoffs": len(rejected),
            "bar_main_temporal_mask_iou_min": temporal_mask_min,
            "bar_main_temporal_bbox_iou_min": temporal_bbox_min,
            "bar_main_centroid_jump_max_normalized": centroid_jump_max,
            "bar_main_area_ratio_max": area_ratio_max,
            "bar_warning_reasons": quality["tracking_warnings"],
            "source_raw_unchanged": int(bool(audit.get("source_raw_unchanged"))),
        })
    return rows


def select_visual_cases(rows):
    selected = defaultdict(set)

    def add(candidates, reason, limit):
        for row in candidates[:limit]:
            selected[row["video_id"]].add(reason)

    handoffs = [row for row in rows if row["bar_accepted_handoffs"]]
    add(handoffs, "accepted_handoff", 2)
    add(sorted(
        [row for row in rows if row["bar_detection_ratio"] < 1.0 - 1e-12],
        key=lambda row: (row["bar_detection_ratio"], row["video_id"]),
    ), "incomplete_detection", 3)
    add(sorted(
        [row for row in rows if row["bar_main_track_coverage"] < 0.80],
        key=lambda row: (row["bar_main_track_coverage"], row["video_id"]),
    ), "low_main_coverage", 5)
    add(sorted(
        [row for row in rows if np.isclose(row["bar_detection_ratio"], 1.0)
         and row["bar_main_track_coverage"] < 0.80],
        key=lambda row: (row["bar_main_track_coverage"], row["video_id"]),
    ), "complete_detection_low_coverage", 3)
    add(sorted(
        [row for row in rows if row["bar_multiple_instance_frames"] > 0],
        key=lambda row: (-row["bar_multiple_instance_frames"], row["video_id"]),
    ), "multiple_bar_instances", 2)
    add(sorted(
        [row for row in rows if row["bar_longest_missing_gap"] > 0],
        key=lambda row: (-row["bar_longest_missing_gap"], row["video_id"]),
    ), "long_detection_gap", 2)
    rejected = sorted(
        [row for row in rows if row["bar_rejected_handoffs"] > 0],
        key=lambda row: (-row["bar_rejected_handoffs"], row["video_id"]),
    )
    add(rejected, "rejected_handoff_candidate", 2)
    controls = [
        row for row in rows
        if np.isclose(row["bar_detection_ratio"], 1.0)
        and np.isclose(row["bar_main_track_coverage"], 1.0)
        and row["bar_number_of_tracks"] == 1
    ]
    add(controls, "perfect_single_track_control", 1)
    multiple_controls = [
        row for row in rows
        if np.isclose(row["bar_main_track_coverage"], 1.0)
        and row["bar_multiple_instance_frames"] > 0
    ]
    add(sorted(
        multiple_controls,
        key=lambda row: (-row["bar_multiple_instance_frames"], row["video_id"]),
    ), "perfect_main_multiple_instances_control", 1)

    ordered = sorted(
        selected,
        key=lambda video_id: (
            next(row["bar_main_track_coverage"] for row in rows if row["video_id"] == video_id),
            video_id,
        ),
    )
    if len(ordered) > 15:
        keep = set(ordered[:15])
        selected = defaultdict(set, {key: value for key, value in selected.items() if key in keep})
    return selected


def bar_frame_data(raw, features):
    class_names = features["class_names"].astype(str).tolist()
    bar_position = class_names.index("bar")
    selected_indices = features["selected_instance_indices"][:, bar_position].astype(int)
    offsets = raw["frame_instance_offsets"].astype(int)
    raw_names = raw["instance_class_name"].astype(str)
    counts = np.asarray([
        np.count_nonzero(raw_names[offsets[index]:offsets[index + 1]] == "bar")
        for index in range(len(offsets) - 1)
    ])
    return selected_indices, counts


def preview_frames(raw, features, audit):
    selected_indices, counts = bar_frame_data(raw, features)
    frames_total = len(selected_indices)
    scenarios = {}
    raw_missing = np.flatnonzero(counts == 0)
    if raw_missing.size:
        scenarios["raw_bar_missing"] = int(raw_missing[0])
    main_missing = np.flatnonzero((counts > 0) & (selected_indices < 0))
    if main_missing.size:
        scenarios["raw_available_main_missing_first"] = int(main_missing[0])
        scenarios["raw_available_main_missing_last"] = int(main_missing[-1])
    multiple = np.flatnonzero(counts > 1)
    if multiple.size:
        scenarios["multiple_bar_instances"] = int(multiple[np.argmax(counts[multiple])])

    bar_identity = audit["merge_audit"]["bar_identity"]
    accepted = bar_identity["accepted_handoffs"]
    if accepted:
        transition = int(accepted[0].get("to_first_frame", 0))
        scenarios["handoff_before"] = max(0, transition - 1)
        scenarios["handoff_after"] = min(frames_total - 1, transition)

    object_names = features["object_feature_names"].astype(str).tolist()
    object_values = features["object_features"][:, features["class_names"].astype(str).tolist().index("bar")]
    dx = object_values[:, object_names.index("centroid_dx_normalized")]
    finite = np.flatnonzero(np.isfinite(dx))
    if finite.size:
        dy = object_values[:, object_names.index("centroid_dy_normalized")]
        jumps = np.hypot(dx[finite], dy[finite])
        scenarios["largest_main_centroid_jump"] = int(finite[np.argmax(jumps)])
    scenarios["sequence_midpoint_control"] = frames_total // 2

    unique = []
    used = set()
    for scenario, frame_index in scenarios.items():
        if frame_index not in used:
            unique.append((scenario, frame_index))
            used.add(frame_index)
    return unique[:6]


def generate_bar_previews(root, dataset_dir, selected):
    preview_root = root / "tracking_v2" / "bar_tracking_audit_previews"
    preview_root.mkdir(parents=True, exist_ok=True)
    visual_rows = []
    for video_id in sorted(selected):
        raw_path = root / "tracking_v2" / "raw" / (video_id + "_segmentation.npz")
        features_path = root / "tracking_v2" / "features" / (video_id + "_features.npz")
        audit_path = root / "tracking_v2" / "audits" / (video_id + "_tracking_audit.json")
        video_path = dataset_dir / (video_id + ".mp4")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise OSError("No se pudo abrir %s" % video_path)
        output_dir = preview_root / video_id
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            with np.load(raw_path, allow_pickle=False) as raw, np.load(features_path, allow_pickle=False) as features:
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                scenarios = preview_frames(raw, features, audit)
                options = default_options()
                options["classes"] = {"bar": True, "bench": False, "person": False}
                options["interactions"] = False
                tiles = []
                saved = []
                for scenario, frame_index in scenarios:
                    frame = read_video_frame(capture, frame_index, raw["frame_image_shape_hw"][frame_index])
                    annotated = annotate_frame(frame, raw, features, frame_index, options)
                    cv2.rectangle(annotated, (0, annotated.shape[0] - 42),
                                  (annotated.shape[1], annotated.shape[0]), (0, 0, 0), -1)
                    cv2.putText(annotated, scenario, (12, annotated.shape[0] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                    path = output_dir / ("frame_%05d_%s.jpg" % (frame_index, scenario))
                    if not cv2.imwrite(str(path), annotated):
                        raise OSError("No se pudo guardar %s" % path)
                    saved.append(frame_index)
                    tiles.append(cv2.resize(annotated, (960, 540)))
                if tiles:
                    while len(tiles) % 2:
                        tiles.append(np.zeros_like(tiles[0]))
                    rows = [np.hstack(tiles[index:index + 2]) for index in range(0, len(tiles), 2)]
                    contact = np.vstack(rows)
                    contact_path = output_dir / (video_id + "_bar_tracking_contact_sheet.jpg")
                    if not cv2.imwrite(str(contact_path), contact):
                        raise OSError("No se pudo guardar %s" % contact_path)
        finally:
            capture.release()
        review = VISUAL_REVIEWS.get(video_id, {})
        visual_rows.append({
            "video_id": video_id,
            "selection_reasons": "|".join(sorted(selected[video_id])),
            "preview_frames": "|".join(map(str, saved)),
            "visual_classification": review.get("classification", "PENDING_REVIEW"),
            "visual_finding": review.get("finding", "Pendiente de revision del contact sheet"),
            "future_temporal_feature_risk": review.get("risk", "PENDING"),
        })
    return visual_rows


def stats_table(grouped):
    lines = [
        "| Metrica | N | Media | Mediana | Std | Min | P1 | P5 | P25 | P50 | P75 | P95 | P99 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in grouped.items():
        value = stats(values)
        p = value["percentiles"]
        lines.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            name, value["count"], fmt(value["mean"]), fmt(value["median"]), fmt(value["std"]),
            fmt(value["min"]), fmt(p["1"]), fmt(p["5"]), fmt(p["25"]), fmt(p["50"]),
            fmt(p["75"]), fmt(p["95"]), fmt(p["99"]), fmt(value["max"]),
        ))
    return lines


def build_report(rows, visual_rows, qa_rows):
    detection = np.asarray([row["bar_detection_ratio"] for row in rows])
    coverage = np.asarray([row["bar_main_track_coverage"] for row in rows])
    counts = {
        "detection_eq_1": int(np.count_nonzero(np.isclose(detection, 1.0))),
        "detection_lt_1": int(np.count_nonzero(detection < 1.0 - 1e-12)),
        "coverage_eq_1": int(np.count_nonzero(np.isclose(coverage, 1.0))),
        "coverage_lt_1": int(np.count_nonzero(coverage < 1.0 - 1e-12)),
        "coverage_lt_095": int(np.count_nonzero(coverage < 0.95)),
        "coverage_lt_080": int(np.count_nonzero(coverage < 0.80)),
        "coverage_lt_060": int(np.count_nonzero(coverage < 0.60)),
    }
    diagnoses = Counter(row["diagnosis"] for row in rows)
    categories = Counter(row["audit_category"] for row in rows)
    complete_incomplete = [
        row for row in rows if row["continuity_pattern"] == "complete_detection_incomplete_main"
    ]
    complete_incomplete_causes = Counter(row["diagnosis"] for row in complete_incomplete)
    qa_person_false = [
        row for row in qa_rows
        if "low_person_detection_ratio" in row["selection_reasons"]
        or "low_person_main_track_coverage" in row["selection_reasons"]
    ]
    grouped = {
        "bar_detection_ratio": detection,
        "bar_main_track_coverage": coverage,
        "bar_confidence_mean_por_video": [row["bar_mean_confidence"] for row in rows],
        "bar_number_of_tracks": [row["bar_number_of_tracks"] for row in rows],
        "bar_longest_missing_gap": [row["bar_longest_missing_gap"] for row in rows],
    }
    lines = [
        "# Reporte BP_FRONT Bar Tracking Audit V1", "",
        "Fecha: `%s`" % utc_now(), "",
        "## 1. Alcance e inmutabilidad", "",
        "- Dataset: `BP_FRONT_SEG_DATASET_V1`; tracking: `bp_front_bench_tracking_v2.2`.",
        "- Videos auditables: **%d**; `bp_349` permanece `QUALITY_ERROR` y no se intento recuperar." % len(rows),
        "- La auditoria solo leyo RAW, tracking V2 y FEATURES; no ejecuto YOLO ni recalculo features.",
        "- Los %d audits confirman `source_raw_unchanged=1`." % sum(row["source_raw_unchanged"] for row in rows),
        "- No se utilizaron labels, resultados ni criterios de jueceo.", "",
        "## 2. Correcciones de metadata y QA", "",
        "- `bp_349` conserva `status=ERROR`, `quality_status=QUALITY_ERROR` y ahora usa `quality_reasons=degenerate_polygon`.",
        "- La seleccion QA ya no marca metricas perfectas como bajas y conserva todos los empates del cutoff.",
        "- Casos falsos `low_person_detection_ratio/main_track_coverage` restantes: **%d**." % len(qa_person_false),
        "- QUALITY_WARNING sigue siendo metadata descriptiva; no implica exclusion.", "",
        "## 3. Conteos reales de BAR", "",
        "| Condicion | Videos |", "|---|---:|",
        "| detection_ratio = 1 | %d |" % counts["detection_eq_1"],
        "| detection_ratio < 1 | %d |" % counts["detection_lt_1"],
        "| main_track_coverage = 1 | %d |" % counts["coverage_eq_1"],
        "| main_track_coverage < 1 | %d |" % counts["coverage_lt_1"],
        "| main_track_coverage < 0.95 | %d |" % counts["coverage_lt_095"],
        "| main_track_coverage < 0.80 | %d |" % counts["coverage_lt_080"],
        "| main_track_coverage < 0.60 | %d |" % counts["coverage_lt_060"], "",
        "## 4. Distribuciones", "",
    ]
    lines += stats_table(grouped)
    lines += ["", "## 5. Diagnostico de continuidad", ""]
    for diagnosis, count in sorted(diagnoses.items(), key=lambda item: (-item[1], item[0])):
        lines.append("- `%s`: **%d** videos." % (diagnosis, count))
    lines += [
        "",
        "Videos con deteccion completa pero principal incompleto: **%d**." % len(complete_incomplete),
        "En estos casos la diferencia no es ausencia YOLO: existen detecciones BAR bajo otros track_id; la auditoria distingue competencia simultanea, fragmentacion y candidatos de handoff rechazados.",
        "", "Desglose de esos casos:", "",
    ]
    for diagnosis, count in sorted(
        complete_incomplete_causes.items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append("- `%s`: **%d** videos." % (diagnosis, count))
    lines += [
        "",
        "Las causas observables mas comunes son cambio de track_id al variar posicion/forma de la barra, competencia entre detecciones simultaneas y handoffs geometricamente plausibles que no superan el criterio V2 estricto. La multiplicidad aislada no implica fallo si el principal conserva cobertura completa.",
        "", "Categorias descriptivas:", "",
    ]
    for category, count in sorted(categories.items()):
        lines.append("- `%s`: **%d** videos." % (category, count))
    lines += [
        "",
        "## 6. Casos estructurales", "",
        "- Videos con multiples instancias BAR: **%d**." % sum(row["bar_multiple_instance_frames"] > 0 for row in rows),
        "- Frames con multiples instancias BAR: **%d**." % sum(row["bar_multiple_instance_frames"] for row in rows),
        "- Videos con mas de un track BAR: **%d**." % sum(row["bar_number_of_tracks"] > 1 for row in rows),
        "- Handoffs BAR aceptados: **%d**." % sum(row["bar_accepted_handoffs"] for row in rows),
        "- Videos con candidatos de handoff rechazados: **%d**." % sum(row["bar_rejected_handoffs"] > 0 for row in rows),
        "",
        "## 7. Revision visual focalizada", "",
        "Se seleccionaron **%d** videos sin labels." % len(visual_rows),
    ]
    for row in visual_rows:
        lines += [
            "### `%s`" % row["video_id"], "",
            "- Seleccion: `%s`." % row["selection_reasons"],
            "- Frames: `%s`." % row["preview_frames"],
            "- Clasificacion visual: `%s`." % row["visual_classification"],
            "- Hallazgo: %s." % row["visual_finding"],
            "- Riesgo para features temporales: `%s`." % row["future_temporal_feature_risk"], "",
        ]
    reviewed = [row for row in visual_rows if row["visual_classification"] != "PENDING_REVIEW"]
    severe = [row for row in reviewed if row["future_temporal_feature_risk"] == "HIGH"]
    conclusion = (
        "Revision visual pendiente; no es posible decidir todavia."
        if not reviewed else
        "La extraccion BP_FRONT_SEG_DATASET_V1 puede permanecer congelada, pero existe evidencia suficiente para justificar Tracking V3 antes de usar la continuidad BAR como feature temporal definitiva."
        if severe else
        "Tracking V2 es suficiente para congelar BP_FRONT_SEG_DATASET_V1; las limitaciones observadas deben mantenerse como metadata descriptiva."
    )
    lines += [
        "## 8. Conclusion tecnica", "", conclusion, "",
        "No se creo Tracking V3 ni se modificaron parametros, asociaciones, mascaras o features.", "",
    ]
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Audita continuidad BAR de BP_FRONT sin recalcular tracking")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--generate-previews", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifests = args.root / "manifests"
    quality_rows = read_csv(manifests / "bp_front_quality_report.csv")
    update_bp349_manifest(args.root)
    qa_rows = update_qa_metadata(args.root, quality_rows)
    audit_rows = build_audit_rows(args.root, quality_rows)
    audit_path = manifests / "bp_front_bar_tracking_audit.csv"
    atomic_write_csv(audit_path, audit_rows, AUDIT_FIELDS)
    selected = select_visual_cases(audit_rows)
    if args.generate_previews:
        visual_rows = generate_bar_previews(args.root, args.dataset_dir, selected)
    else:
        visual_rows = [
            {
                "video_id": video_id,
                "selection_reasons": "|".join(sorted(reasons)),
                "preview_frames": "",
                "visual_classification": VISUAL_REVIEWS.get(video_id, {}).get("classification", "PENDING_REVIEW"),
                "visual_finding": VISUAL_REVIEWS.get(video_id, {}).get("finding", "Pendiente de revision"),
                "future_temporal_feature_risk": VISUAL_REVIEWS.get(video_id, {}).get("risk", "PENDING"),
            }
            for video_id, reasons in sorted(selected.items())
        ]
    visual_path = manifests / "bp_front_bar_tracking_visual_review.csv"
    atomic_write_csv(visual_path, visual_rows, VISUAL_FIELDS)
    atomic_write_text(args.report, build_report(audit_rows, visual_rows, qa_rows))
    print("Auditoria BAR: %s" % audit_path)
    print("Seleccion visual: %s" % visual_path)
    print("Reporte: %s" % args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
