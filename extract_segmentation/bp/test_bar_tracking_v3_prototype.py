import csv
import json
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
ROOT = PROJECT_ROOT / "segmentation" / "bp" / "front"
OUTPUT = ROOT / "tracking_v3" / "test_14"

EXPECTED_PATHS = {
    "bp_001": "0",
    "bp_004": "0|3|4",
    "bp_019": "0|4",
    "bp_067": "0|3",
    "bp_139": "0",
    "bp_198": "0|5|6",
    "bp_241": "0",
    "bp_301": "0|4",
    "bp_467": "0|4",
    "bp_469": "0",
    "bp_518": "0",
    "bp_520": "7",
    "bp_541": "0|5",
    "bp_544": "0|5",
}

EXPECTED_IMPROVED = {
    "bp_004", "bp_019", "bp_067", "bp_198",
    "bp_301", "bp_467", "bp_541", "bp_544",
}

EXPECTED_CONTROLS = {"bp_001", "bp_518", "bp_520"}


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main():
    comparisons = read_csv(
        OUTPUT / "manifests" / "bp_front_tracking_v3_prototype_comparison.csv"
    )
    if len(comparisons) != 14:
        raise AssertionError("Se esperaban 14 comparaciones")
    by_video = {row["video_id"]: row for row in comparisons}
    if set(by_video) != set(EXPECTED_PATHS):
        raise AssertionError("La selección de videos cambió")

    for video_id, expected_path in EXPECTED_PATHS.items():
        row = by_video[video_id]
        if row["v3_tracklet_path"] != expected_path:
            raise AssertionError("Path inesperado para %s" % video_id)
        if row["uses_yolo"] != "0" or row["uses_labels"] != "0":
            raise AssertionError("V3 declaró YOLO o labels en %s" % video_id)
        if float(row["v3_main_track_coverage"]) < float(row["v2_main_track_coverage"]):
            raise AssertionError("Regresión de coverage en %s" % video_id)
        expected_class = "IMPROVED" if video_id in EXPECTED_IMPROVED else "UNCHANGED"
        if row["classification"] != expected_class:
            raise AssertionError("Clasificación inesperada en %s" % video_id)

        selection_path = OUTPUT / "selections" / (video_id + "_bar_tracking_v3_selection.npz")
        audit_path = OUTPUT / "audits" / (video_id + "_bar_tracking_v3_audit.json")
        with np.load(selection_path, allow_pickle=False) as selection:
            selected = selection["selected_instance_indices"].astype(int)
            metadata = json.loads(str(selection["metadata_json"]))
            if metadata["uses_yolo"] or metadata["uses_labels"] or metadata["interpolation"]:
                raise AssertionError("Metadata metodológica inválida en %s" % video_id)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit["source_raw_unchanged"] or not audit["source_v2_unchanged"]:
            raise AssertionError("Integridad fuente no confirmada en %s" % video_id)
        if any(selected[frame] >= 0 for frame in audit["raw_detection_gap_frames"]):
            raise AssertionError("V3 inventó detección en gap RAW de %s" % video_id)
        if int(np.count_nonzero(selected >= 0)) != int(audit["v3_selected_frames"]):
            raise AssertionError("Coverage NPZ/audit inconsistente en %s" % video_id)

        if video_id in EXPECTED_CONTROLS:
            features_path = ROOT / "tracking_v2" / "features" / (video_id + "_features.npz")
            with np.load(features_path, allow_pickle=False) as features:
                position = features["class_names"].astype(str).tolist().index("bar")
                v2_indices = features["selected_instance_indices"][:, position].astype(int)
            if not np.array_equal(selected, v2_indices):
                raise AssertionError("Control V2 degradado en %s" % video_id)

    print("Tracking V3 prototype tests: OK (14/14)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
