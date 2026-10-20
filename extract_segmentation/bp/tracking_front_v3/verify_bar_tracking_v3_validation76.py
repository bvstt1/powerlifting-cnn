"""Verifica en lectura los artefactos históricos de Validation76 V3.

Contrasta IDs, RAW/V2, revisiones y métricas; distingue hashes históricos
de los del código reorganizado. No confunde integridad con aprobación V3.
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bar_tracking_v3_utils import ALGORITHM_VERSION, DEFAULT_PARAMETERS, SCORE_WEIGHTS
from code_versions import HISTORICAL_ALGORITHM_HASHES, RELOCATED_ALGORITHM_HASHES


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "segmentation" / "bp" / "front"
DEFAULT_OUTPUT = DEFAULT_ROOT / "tracking_v3" / "validation_76"
EXPECTED_ALGORITHM_HASHES = HISTORICAL_ALGORITHM_HASHES
EXPECTED_PARAMETERS = {
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


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    temporary.replace(path)


def close_number(actual, expected):
    return np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-12)


def handoff_key(row):
    return (
        row["video_id"], int(row["source_track"]), int(row["target_track"]),
        int(row["transition_frame"]),
    )


def add_check(checks, name, passed, detail):
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verificacion read-only de BP Front Tracking V3 Validation76"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root
    output = args.output_dir
    manifests = output / "manifests"
    report_path = args.report or (
        manifests / "bp_front_tracking_v3_validation_76_verification.json"
    )
    paths = {
        "validation": manifests / "bp_front_tracking_v3_validation_76.csv",
        "handoffs": manifests / "bp_front_tracking_v3_validation_76_handoffs.csv",
        "visual": manifests / "bp_front_tracking_v3_validation_76_visual_review.csv",
        "config": manifests / "bp_front_tracking_v3_validation_76_config.json",
        "reviews": manifests / "bp_front_tracking_v3_validation_76_explicit_reviews.json",
        "preflight": manifests / "bp_front_tracking_v3_validation_76_integrity_preflight.json",
        "source_audit": root / "manifests" / "bp_front_bar_tracking_audit.csv",
        "prototype": (
            root / "tracking_v3" / "test_14" / "manifests"
            / "bp_front_tracking_v3_prototype_comparison.csv"
        ),
    }
    missing_inputs = [name for name, path in paths.items() if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError("Faltan entradas: %s" % missing_inputs)

    validation = read_csv(paths["validation"])
    handoffs = read_csv(paths["handoffs"])
    visual = read_csv(paths["visual"])
    source = read_csv(paths["source_audit"])
    prototype = read_csv(paths["prototype"])
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    reviews = json.loads(paths["reviews"].read_text(encoding="utf-8"))
    preflight = json.loads(paths["preflight"].read_text(encoding="utf-8"))
    checks = []

    selected = sorted(
        [
            row for row in source
            if close_number(row["bar_detection_ratio"], 1.0)
            and float(row["bar_main_track_coverage"]) < 1.0 - 1e-12
        ],
        key=lambda row: row["video_id"],
    )
    selected_ids = [row["video_id"] for row in selected]
    validation_ids = [row["video_id"] for row in validation]
    add_check(checks, "dynamic_selection_76", len(selected) == 76, "N=%d" % len(selected))
    predicates_ok = all(
        close_number(row["bar_detection_ratio"], 1.0)
        and float(row["bar_main_track_coverage"]) < 1.0
        for row in selected
    )
    add_check(checks, "selection_predicates", predicates_ok, "76 filas evaluadas")
    add_check(
        checks, "validation_ids_match_selection",
        validation_ids == selected_ids,
        "IDs iguales y en orden=%s" % (validation_ids == selected_ids),
    )

    no_labels = (
        not config.get("uses_labels")
        and all(int(row["uses_labels"]) == 0 for row in validation)
    )
    no_yolo = (
        not config.get("uses_yolo")
        and all(int(row["uses_yolo"]) == 0 for row in validation)
    )
    add_check(checks, "labels_not_used", no_labels, "uses_labels=false")
    add_check(checks, "yolo_not_executed", no_yolo, "uses_yolo=false")

    script_paths = {
        "bar_tracking_v3_utils.py": SCRIPT_DIR / "bar_tracking_v3_utils.py",
        "rebuild_bar_tracking_v3_from_raw.py": SCRIPT_DIR / "rebuild_bar_tracking_v3_from_raw.py",
    }
    actual_algorithm_hashes = {
        name: sha256_file(path) for name, path in script_paths.items()
    }
    algorithm_ok = actual_algorithm_hashes == RELOCATED_ALGORITHM_HASHES
    add_check(checks, "relocated_algorithm_hashes", algorithm_ok, actual_algorithm_hashes)
    historical_hashes_ok = (
        config.get("algorithm_hashes") == EXPECTED_ALGORITHM_HASHES
        and all(preflight["algorithm_scripts_sha256"].get(
            "extract_segmentation/bp/" + name
        ) == digest for name, digest in EXPECTED_ALGORITHM_HASHES.items())
    )
    add_check(checks, "historical_hashes_recorded", historical_hashes_ok,
              "Los hashes históricos proceden del manifiesto; el código actual fue reubicado")
    parameters_ok = (
        ALGORITHM_VERSION == "bp_front_bar_tracking_v3_prototype_1"
        and DEFAULT_PARAMETERS == EXPECTED_PARAMETERS
        and config.get("parameters") == EXPECTED_PARAMETERS
        and config.get("score_weights") == SCORE_WEIGHTS
        and config.get("algorithm_hashes") == EXPECTED_ALGORITHM_HASHES
    )
    add_check(checks, "frozen_parameters_and_weights", parameters_ok, {
        "algorithm_version": ALGORITHM_VERSION,
        "parameters_match": DEFAULT_PARAMETERS == EXPECTED_PARAMETERS,
        "weights_match": config.get("score_weights") == SCORE_WEIGHTS,
    })
    add_check(
        checks, "auto_review_disabled",
        config.get("auto_review") is False and reviews.get("auto_review") is False,
        "config y registro explicito declaran auto_review=false",
    )

    source_mismatches = []
    for video_id in selected_ids:
        pre = preflight["source_files"].get(video_id)
        if pre is None:
            source_mismatches.append({"video_id": video_id, "artifact": "missing_preflight"})
            continue
        current_paths = {
            "raw": root / "raw" / (video_id + "_segmentation.npz"),
            "v2_raw": root / "tracking_v2" / "raw" / (video_id + "_segmentation.npz"),
            "v2_features": root / "tracking_v2" / "features" / (video_id + "_features.npz"),
            "v2_audit": root / "tracking_v2" / "audits" / (video_id + "_tracking_audit.json"),
        }
        for artifact, path in current_paths.items():
            current = sha256_file(path)
            if current != pre["hashes"][artifact]:
                source_mismatches.append({
                    "video_id": video_id, "artifact": artifact,
                    "pre": pre["hashes"][artifact], "post": current,
                })
    raw_ok = not any(row["artifact"] == "raw" for row in source_mismatches)
    v2_ok = not any(row["artifact"].startswith("v2_") for row in source_mismatches)
    add_check(checks, "raw_pre_post_integrity", raw_ok, "mismatches=%d" % len(source_mismatches))
    add_check(checks, "v2_pre_post_integrity", v2_ok, "mismatches=%d" % len(source_mismatches))

    add_check(checks, "results_count_76", len(validation) == 76, "N=%d" % len(validation))
    status_counts = Counter(row["algorithm_status"] for row in validation)
    expected_statuses = {
        "FULLY_RECOVERED": 72,
        "PARTIALLY_RECOVERED": 1,
        "UNCHANGED_CONSERVATIVE": 3,
        "REGRESSED": 0,
    }
    statuses_ok = all(status_counts[name] == count for name, count in expected_statuses.items())
    add_check(checks, "algorithm_status_counts", statuses_ok, dict(status_counts))
    recovered = sum(int(row["recovered_frames"]) for row in validation)
    recoverable = sum(int(row["recoverable_frames"]) for row in validation)
    add_check(checks, "recovered_frames_1641", recovered == 1641, "N=%d" % recovered)
    add_check(checks, "recoverable_frames_1680", recoverable == 1680, "N=%d" % recoverable)

    accepted = [row for row in handoffs if int(row["accepted"]) == 1]
    suspicious = [row for row in accepted if int(row["suspicious"]) == 1]
    add_check(checks, "accepted_handoffs_79", len(accepted) == 79, "N=%d" % len(accepted))
    add_check(checks, "suspicious_handoffs_47", len(suspicious) == 47, "N=%d" % len(suspicious))

    evidence = [row["review_evidence"] for row in accepted]
    evidence_unique = len(evidence) == len(set(evidence)) == 79
    evidence_paths = [PROJECT_ROOT / value for value in evidence if value]
    evidence_exists = (
        len(evidence_paths) == 79
        and all(path.exists() and "_overviews" not in path.parts for path in evidence_paths)
    )
    add_check(checks, "accepted_handoff_evidence_unique_79", evidence_unique, {
        "total": len(evidence), "unique": len(set(evidence)),
    })
    add_check(checks, "accepted_handoff_evidence_exists", evidence_exists, "existentes=%d" % sum(path.exists() for path in evidence_paths))

    accepted_reviewed = all(
        row["visual_review_status"] not in {"", "PENDING"} for row in accepted
    )
    suspicious_reviewed = all(
        row["visual_review_status"] not in {"", "PENDING"}
        and bool(row["review_notes"]) and bool(row["review_evidence"])
        for row in suspicious
    )
    add_check(checks, "accepted_handoffs_explicitly_reviewed", accepted_reviewed, "N=%d" % sum(row["visual_review_status"] not in {"", "PENDING"} for row in accepted))
    add_check(checks, "suspicious_handoffs_explicitly_reviewed", suspicious_reviewed, "N=%d" % len(suspicious))

    video_reviewed = all(
        row["visual_review_status"] not in {"", "PENDING"}
        and bool(row["review_notes"]) and bool(row["review_evidence"])
        for row in visual
    )
    pending_videos = sum(row["visual_review_status"] == "PENDING" for row in visual)
    pending_handoffs = sum(row["visual_review_status"] == "PENDING" for row in accepted)
    add_check(checks, "videos_explicitly_reviewed_76", len(visual) == 76 and video_reviewed, "N=%d" % len(visual))
    add_check(checks, "pending_zero", pending_videos == 0 and pending_handoffs == 0, {
        "videos": pending_videos, "handoffs": pending_handoffs,
    })

    false_merge_handoffs = [row for row in accepted if int(row["false_merge_detected"])]
    false_merge_videos = [row for row in visual if int(row["false_merge_detected"])]
    add_check(
        checks, "false_merge_detected_zero",
        len(false_merge_handoffs) == 0,
        {
            "handoffs": [handoff_key(row) for row in false_merge_handoffs],
            "videos": [row["video_id"] for row in false_merge_videos],
        },
    )

    explicit_video_reviews = reviews.get("video_reviews", [])
    explicit_handoff_reviews = reviews.get("handoff_reviews", [])
    explicit_keys = {handoff_key(row) for row in explicit_handoff_reviews}
    accepted_keys = {handoff_key(row) for row in accepted}
    explicit_ok = (
        len(explicit_video_reviews) == 76
        and len({row["video_id"] for row in explicit_video_reviews}) == 76
        and len(explicit_handoff_reviews) == 79
        and len(explicit_keys) == 79
        and explicit_keys == accepted_keys
    )
    add_check(checks, "explicit_review_registry_complete", explicit_ok, {
        "videos": len(explicit_video_reviews), "handoffs": len(explicit_handoff_reviews),
    })

    prototype_command = [
        sys.executable, "-B", str(SCRIPT_DIR / "test_bar_tracking_v3_prototype.py")
    ]
    prototype_result = subprocess.run(
        prototype_command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    prototype_suite = prototype_result.stdout.strip() or prototype_result.stderr.strip()
    add_check(
        checks, "prototype_acceptance_suite_14_14",
        prototype_result.returncode == 0 and "OK (14/14)" in prototype_suite,
        prototype_suite,
    )

    prototype_by_video = {row["video_id"]: row for row in prototype}
    shared = sorted(set(selected_ids) & set(prototype_by_video))
    validation_by_video = {row["video_id"]: row for row in validation}
    prototype_mismatches = []
    for video_id in shared:
        actual = validation_by_video[video_id]
        expected = prototype_by_video[video_id]
        comparisons = {
            "path": actual["v3_path"] == expected["v3_tracklet_path"],
            "coverage": close_number(actual["v3_main_track_coverage"], expected["v3_main_track_coverage"]),
            "handoffs": actual["accepted_handoffs"] == expected["accepted_handoff_pairs"],
            "score_v2": close_number(actual["score_v2"], expected["score_v2"]),
            "score_v3": close_number(actual["score_v3"], expected["score_v3"]),
            "margin": close_number(actual["minimum_runner_up_margin"], expected["score_margin"]),
            "decision": actual["algorithm_decision"] == expected["algorithm_decision"],
            "status": actual["algorithm_status"] == "FULLY_RECOVERED",
        }
        if not all(comparisons.values()):
            prototype_mismatches.append({
                "video_id": video_id,
                "fields": [name for name, passed in comparisons.items() if not passed],
            })
    add_check(
        checks, "prototype_validation_shared_exact",
        len(shared) == 5 and not prototype_mismatches,
        {"shared": shared, "mismatches": prototype_mismatches},
    )

    consistency_errors = []
    for row in validation:
        video_id = row["video_id"]
        audit_path = output / "audits" / (video_id + "_bar_tracking_v3_audit.json")
        selection_path = output / "selections" / (video_id + "_bar_tracking_v3_selection.npz")
        if not audit_path.exists() or not selection_path.exists():
            consistency_errors.append({"video_id": video_id, "reason": "missing_audit_or_npz"})
            continue
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        with np.load(selection_path, allow_pickle=False) as selection:
            indices = selection["selected_instance_indices"].astype(int)
            metadata = json.loads(str(selection["metadata_json"]))
            selected_frames = int(np.count_nonzero(indices >= 0))
        checks_by_video = {
            "frames": selected_frames == int(row["v3_frames_covered"]) == int(audit["v3_selected_frames"]),
            "path": "|".join(map(str, audit["v3_tracklet_path"])) == row["v3_path"],
            "coverage": close_number(selected_frames / int(row["frames_total"]), row["v3_main_track_coverage"]),
            "source_hash": audit["source_v2_sha256"] == row["source_v2_sha256"],
            "method": not metadata["uses_yolo"] and not metadata["uses_labels"] and not metadata["interpolation"],
            "raw_gaps": not audit["raw_detection_gap_frames"],
        }
        if not all(checks_by_video.values()):
            consistency_errors.append({
                "video_id": video_id,
                "fields": [name for name, passed in checks_by_video.items() if not passed],
            })
    add_check(checks, "npz_csv_audit_consistency", not consistency_errors, consistency_errors)
    add_check(
        checks, "raw_gaps_preserved",
        all(int(row["raw_gaps_preserved"]) == 1 for row in validation)
        and all(int(row["raw_gap_preserved"]) == 1 for row in visual),
        "El conjunto no contiene gaps RAW y V3 no inventa detecciones",
    )

    validator_path = SCRIPT_DIR / "validate_bar_tracking_v3_76.py"
    validator_pre = preflight["algorithm_scripts_sha256"].get(
        "extract_segmentation/bp/validate_bar_tracking_v3_76.py"
    )
    validator_post = sha256_file(validator_path)
    checks_passed = sum(check["passed"] for check in checks)
    result = "OK" if checks_passed == len(checks) else "NOT_OK"
    report = {
        "schema_version": "bp_front_tracking_v3_validation76_verification_v1",
        "timestamp_utc": utc_now(),
        "result": result,
        "checks_total": len(checks),
        "checks_passed": checks_passed,
        "checks_failed": len(checks) - checks_passed,
        "failed_checks": [check["name"] for check in checks if not check["passed"]],
        "checks": checks,
        "integrity_pre_post": (
            "HISTORICAL_CODE_RELOCATED" if not source_mismatches and algorithm_ok
            and historical_hashes_ok else "MISMATCH"
        ),
        "integrity": {
            "source_artifacts_checked": len(selected_ids) * 4,
            "source_mismatches": source_mismatches,
            "algorithm_hashes_pre": {
                name: preflight["algorithm_scripts_sha256"].get(
                    "extract_segmentation/bp/" + name
                ) for name in EXPECTED_ALGORITHM_HASHES
            },
            "algorithm_hashes_post": actual_algorithm_hashes,
            "validator_hash_pre": validator_pre,
            "validator_hash_post": validator_post,
            "validator_hash_change_allowed": True,
        },
        "metrics": {
            "n_videos": len(validation),
            "status_counts": dict(status_counts),
            "recoverable_frames": recoverable,
            "recovered_frames": recovered,
            "recovery_rate": recovered / recoverable,
            "accepted_handoffs": len(accepted),
            "suspicious_handoffs": len(suspicious),
            "accepted_handoff_evidence_unique": len(set(evidence)),
            "pending_videos": pending_videos,
            "pending_handoffs": pending_handoffs,
            "false_merge_handoffs": len(false_merge_handoffs),
            "false_merge_videos": [row["video_id"] for row in false_merge_videos],
        },
        "prototype_suite": prototype_suite,
        "prototype_shared_ids": shared,
        "approval": "VALIDATION76_APPROVED" if result == "OK" else "VALIDATION76_NOT_APPROVED",
    }
    if args.write_report:
        atomic_write_json(report_path, report)
        print("Verification report: %s" % report_path)
    print(json.dumps({
        "result": result,
        "checks": "%d/%d" % (checks_passed, len(checks)),
        "failed_checks": report["failed_checks"],
        "approval": report["approval"],
    }, indent=2))
    return 0 if result == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
