import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
UTILS_DIR = PROJECT_ROOT / "extract_segmentation" / "bp"
sys.path.insert(0, str(UTILS_DIR))

from segmentation_utils import (  # noqa: E402
    compute_mask_interaction,
    minimum_mask_distance,
    reconstruct_mask,
    sha256_file,
    validate_features_file,
    validate_raw_file,
)


DEFAULT_SEGMENTATION = PROJECT_ROOT / "segmentation" / "bp" / "front"
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "bp" / "front"
COLORS = {
    "bar": (0, 255, 255),
    "bench": (255, 120, 0),
    "person": (0, 220, 0),
}


def normalize_video_id(value):
    stem = Path(value).stem
    if stem.isdigit():
        return "bp_%03d" % int(stem)
    return stem


def feature_indices(names):
    return {name: index for index, name in enumerate(np.asarray(names).astype(str))}


def alpha_mask(frame, mask, color, alpha=0.35):
    if not np.any(mask):
        return
    color_layer = np.empty_like(frame)
    color_layer[:] = color
    frame[mask] = cv2.addWeighted(frame[mask], 1.0 - alpha, color_layer[mask], alpha, 0)


def get_frame_instances(raw, frame_index):
    start = int(raw["frame_instance_offsets"][frame_index])
    end = int(raw["frame_instance_offsets"][frame_index + 1])
    return range(start, end)


def selected_masks_for_frame(raw, features, frame_index):
    result = {}
    class_names = features["class_names"].astype(str)
    selected = features["selected_instance_indices"][frame_index]
    for position, class_name in enumerate(class_names):
        instance_index = int(selected[position])
        if instance_index >= 0:
            result[class_name] = (
                reconstruct_mask(raw, instance_index),
                instance_index,
            )
    return result


def draw_interaction_line(frame, mask_a, mask_b, color):
    distance, point_a, point_b = minimum_mask_distance(
        mask_a, mask_b, return_points=True
    )
    if point_a is not None and point_b is not None:
        cv2.line(frame, point_a, point_b, color, 2, cv2.LINE_AA)
        cv2.circle(frame, point_a, 4, color, -1)
        cv2.circle(frame, point_b, 4, color, -1)
    return distance


def annotate_frame(frame, raw, features, frame_index, options):
    output = frame.copy()
    selected_track_ids = {
        class_name: int(track_id)
        for class_name, track_id in zip(
            features["class_names"].astype(str),
            features["selected_track_ids"],
        )
    }
    object_names = feature_indices(features["object_feature_names"])
    interaction_names = feature_indices(features["interaction_feature_names"])
    pair_names = features["interaction_pair_names"].astype(str)

    for instance_index in get_frame_instances(raw, frame_index):
        class_name = str(raw["instance_class_name"][instance_index])
        if not options["classes"].get(class_name, True):
            continue
        color = COLORS.get(class_name, (255, 255, 255))
        track_id = int(raw["instance_track_id"][instance_index])
        is_main = track_id == selected_track_ids.get(class_name, -1)
        mask = reconstruct_mask(raw, instance_index)
        if options["masks"]:
            alpha_mask(output, mask, color, 0.42 if is_main else 0.20)

        x1, y1, x2, y2 = raw["instance_bbox_xyxy_px"][instance_index].astype(int)
        if options["boxes"]:
            thickness = 4 if is_main and options["main"] else 2
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        ys, xs = np.nonzero(mask)
        if options["centroids"] and xs.size:
            centroid = (int(round(xs.mean())), int(round(ys.mean())))
            cv2.drawMarker(output, centroid, color, cv2.MARKER_CROSS, 18, 2)

        labels = []
        if options["tracks"]:
            labels.append("%s track=%d%s" % (
                class_name,
                track_id,
                " MAIN" if is_main and options["main"] else "",
            ))
        if options["confidence"]:
            labels.append("conf=%.3f" % float(raw["instance_confidence"][instance_index]))
        if labels:
            cv2.putText(
                output,
                " ".join(labels),
                (max(0, x1), max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                color,
                2,
                cv2.LINE_AA,
            )

    text_lines = [
        "frame=%d time=%.3fs" % (frame_index, float(features["timestamp_sec"][frame_index]))
    ]
    if options["interactions"]:
        selected_masks = selected_masks_for_frame(raw, features, frame_index)
        pair_colors = [(255, 0, 255), (0, 128, 255), (255, 255, 255)]
        for pair_index, pair_name in enumerate(pair_names):
            name_a, name_b = pair_name.split("_", 1)
            if not options["classes"].get(name_a, True) or not options["classes"].get(name_b, True):
                continue
            values = features["interaction_features"][frame_index, pair_index]
            both = values[interaction_names["both_detected"]] == 1.0
            if not both:
                text_lines.append("%s: missing" % pair_name)
                continue
            if name_a in selected_masks and name_b in selected_masks:
                mask_a = selected_masks[name_a][0]
                mask_b = selected_masks[name_b][0]
                displayed_distance = draw_interaction_line(
                    output,
                    mask_a,
                    mask_b,
                    pair_colors[pair_index],
                )
                ys_a, xs_a = np.nonzero(mask_a)
                ys_b, xs_b = np.nonzero(mask_b)
                recalculated = compute_mask_interaction(
                    mask_a,
                    mask_b,
                    (float(xs_a.mean()), float(ys_a.mean())),
                    (float(xs_b.mean()), float(ys_b.mean())),
                    options["proximity_threshold"],
                )
            else:
                displayed_distance = np.nan
                recalculated = None
            stored_distance = values[interaction_names["minimum_mask_distance_px"]]
            iou = values[interaction_names["iou"]]
            overlap = int(values[interaction_names["overlap_detected"]])
            proximity = int(values[interaction_names["proximity_candidate"]])
            mismatch = recalculated is None or not (
                np.isclose(stored_distance, recalculated["minimum_mask_distance_px"], atol=1e-5)
                and np.isclose(iou, recalculated["iou"], atol=1e-9)
                and overlap == int(recalculated["overlap_detected"])
                and proximity == int(recalculated["proximity_candidate"])
            )
            text_lines.append(
                "%s%s: d=%.2fpx draw=%.2f IoU=%.4f overlap=%d prox=%d"
                % (
                    "WARNING " if mismatch else "",
                    pair_name,
                    stored_distance,
                    displayed_distance,
                    iou,
                    overlap,
                    proximity,
                )
            )

    if options["main"]:
        detected_index = object_names["detected"]
        detected_main = []
        for position, class_name in enumerate(features["class_names"].astype(str)):
            if features["object_features"][frame_index, position, detected_index] == 1.0:
                detected_main.append("%s:%d" % (class_name, selected_track_ids[class_name]))
        text_lines.append("main " + " ".join(detected_main))

    panel_height = 30 + 25 * len(text_lines)
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (output.shape[1], panel_height), (0, 0, 0), -1)
    output[:panel_height] = cv2.addWeighted(
        output[:panel_height], 0.25, overlay[:panel_height], 0.75, 0
    )
    for line_index, line in enumerate(text_lines):
        cv2.putText(
            output,
            line,
            (12, 25 + 25 * line_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output


def choose_preview_frames(raw, features):
    interaction_names = feature_indices(features["interaction_feature_names"])
    pair_names = features["interaction_pair_names"].astype(str).tolist()
    object_names = feature_indices(features["object_feature_names"])
    scenarios = {}

    if "bar_person" in pair_names:
        pair_index = pair_names.index("bar_person")
        values = features["interaction_features"][:, pair_index]
        both = values[:, interaction_names["both_detected"]] == 1.0
        distance = values[:, interaction_names["minimum_mask_distance_px"]]
        intersection = values[:, interaction_names["intersection_area_px"]]
        valid = np.flatnonzero(both & np.isfinite(distance))
        if valid.size:
            scenarios["bar_person_max_distance"] = int(valid[np.argmax(distance[valid])])
            positive = valid[distance[valid] > 0]
            if positive.size:
                scenarios["bar_person_near"] = int(positive[np.argmin(distance[positive])])
            overlap = valid[intersection[valid] > 0]
            if overlap.size:
                scenarios["bar_person_overlap"] = int(overlap[np.argmax(intersection[overlap])])

    if "person_bench" in pair_names:
        pair_index = pair_names.index("person_bench")
        values = features["interaction_features"][:, pair_index]
        both = values[:, interaction_names["both_detected"]] == 1.0
        iou = values[:, interaction_names["iou"]]
        valid = np.flatnonzero(both & np.isfinite(iou))
        if valid.size:
            scenarios["person_bench_relation"] = int(valid[np.argmax(iou[valid])])

    person_class_ids = {
        int(class_id) for class_id, name in zip(raw["instance_class_id"], raw["instance_class_name"].astype(str))
        if name == "person"
    }
    person_id = next(iter(person_class_ids), None)
    if person_id is not None:
        for frame_index in range(len(raw["frame_index"])):
            instances = list(get_frame_instances(raw, frame_index))
            count = sum(int(raw["instance_class_id"][index]) == person_id for index in instances)
            if count > 1:
                scenarios["multiple_person"] = frame_index
                break

    track_starts = {}
    for instance_index, track_id in enumerate(raw["instance_track_id"].astype(int)):
        track_starts.setdefault(track_id, int(raw["instance_frame_index"][instance_index]))
    secondary_starts = [frame for track_id, frame in track_starts.items() if frame > 0]
    if secondary_starts:
        scenarios["track_change_or_new_track"] = min(secondary_starts)

    detected_index = object_names["detected"]
    detected = features["object_features"][:, :, detected_index]
    missing_frames = np.flatnonzero(np.any(detected == 0.0, axis=1))
    if missing_frames.size:
        scenarios["missing_main_detection"] = int(missing_frames[0])

    confidence_index = object_names["confidence"]
    confidences = features["object_features"][:, :, confidence_index]
    finite_rows = np.flatnonzero(np.any(np.isfinite(confidences), axis=1))
    if finite_rows.size:
        row_minimum = np.nanmin(confidences[finite_rows], axis=1)
        scenarios["lowest_main_confidence"] = int(finite_rows[np.argmin(row_minimum)])

    unique = {}
    used = set()
    for scenario, frame_index in scenarios.items():
        if frame_index not in used:
            unique[scenario] = frame_index
            used.add(frame_index)
    return unique


def read_video_frame(capture, frame_index, expected_shape=None):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise ValueError("No se pudo decodificar frame %d" % frame_index)
    if expected_shape is not None and frame.shape[:2] != tuple(expected_shape):
        expected_height, expected_width = map(int, expected_shape)
        is_codec_padding = (
            frame.shape[1] == expected_width
            and expected_height < frame.shape[0] <= expected_height + 16
        )
        if is_codec_padding:
            frame = frame[:expected_height, :expected_width]
        else:
            raise ValueError(
                "Frame %d tiene shape %s, incompatible con la geometría RAW %s"
                % (frame_index, frame.shape[:2], tuple(expected_shape))
            )
    return frame


def default_options():
    return {
        "classes": {"bar": True, "bench": True, "person": True},
        "masks": True,
        "boxes": True,
        "centroids": True,
        "tracks": True,
        "main": True,
        "confidence": True,
        "interactions": True,
        "proximity_threshold": 0.01,
    }


def save_previews(video_id, capture, raw, features, preview_root, options, max_previews):
    scenarios = choose_preview_frames(raw, features)
    output_dir = preview_root / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    expected_shape = raw["frame_image_shape_hw"][0]
    for scenario, frame_index in list(scenarios.items())[:max_previews]:
        frame = read_video_frame(capture, frame_index, expected_shape)
        annotated = annotate_frame(frame, raw, features, frame_index, options)
        path = output_dir / ("frame_%05d_%s.jpg" % (frame_index, scenario))
        if not cv2.imwrite(str(path), annotated):
            raise OSError("No se pudo guardar preview %s" % path)
        saved.append((scenario, frame_index, path))
    return saved


def parse_args():
    parser = argparse.ArgumentParser(description="Inspector visual de segmentación BP_FRONT")
    parser.add_argument("--video", required=True, help="ID bp_XXX o ruta al video")
    parser.add_argument("--segmentation-dir", type=Path, default=DEFAULT_SEGMENTATION)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--save-previews", action="store_true")
    parser.add_argument("--max-previews", type=int, default=8)
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    video_id = normalize_video_id(args.video)
    video_path = Path(args.video)
    if not video_path.exists():
        video_path = args.dataset_dir / (video_id + ".mp4")
    raw_path = args.segmentation_dir / "raw" / (video_id + "_segmentation.npz")
    features_path = args.segmentation_dir / "features" / (video_id + "_features.npz")
    preview_root = args.segmentation_dir / "previews"

    validate_raw_file(raw_path)
    validate_features_file(features_path, raw_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError("No se pudo abrir %s" % video_path)
    options = default_options()

    try:
        with np.load(raw_path, allow_pickle=False) as raw, np.load(features_path, allow_pickle=False) as features:
            metadata = json.loads(str(raw["metadata_json"]))
            if metadata["video_id"] != video_id:
                raise ValueError("El video no corresponde a los NPZ seleccionados")
            if sha256_file(video_path) != metadata["video_sha256"]:
                raise ValueError("El hash del video no corresponde a la extracción RAW")
            if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != int(metadata["frames_video"]):
                raise ValueError("Cantidad de frames del video distinta a metadata")
            if int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) != int(metadata["width"]):
                raise ValueError("Ancho del video distinto a metadata")
            if int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) != int(metadata["height"]):
                raise ValueError("Alto del video distinto a metadata")
            if not np.isclose(capture.get(cv2.CAP_PROP_FPS), float(metadata["fps"])):
                raise ValueError("FPS del video distinto a metadata")
            options["proximity_threshold"] = float(metadata["proximity_threshold_normalized"])
            frames_total = len(raw["frame_index"])
            if frames_total == 0:
                raise ValueError("No hay frames para inspeccionar")
            if args.save_previews:
                saved = save_previews(
                    video_id, capture, raw, features, preview_root,
                    options, args.max_previews,
                )
                for scenario, frame_index, path in saved:
                    print("[PREVIEW] %s frame=%d -> %s" % (scenario, frame_index, path))

            if args.no_display:
                return 0

            frame_index = min(max(args.frame, 0), frames_total - 1)
            paused = True
            window = "BP_FRONT segmentation inspector"
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 1280, 720)
            print("Teclas: q salir, espacio play/pause, a/d frame, 1/2/3 clases")
            print("m masks, b boxes, c centroides, t tracks, p principal")
            print("f confidence, i interacciones")
            while True:
                frame = read_video_frame(
                    capture, frame_index, raw["frame_image_shape_hw"][frame_index]
                )
                annotated = annotate_frame(frame, raw, features, frame_index, options)
                cv2.imshow(window, annotated)
                key = cv2.waitKey(0 if paused else 33) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    paused = not paused
                elif key == ord("a"):
                    frame_index = max(0, frame_index - 1)
                    paused = True
                elif key == ord("d"):
                    frame_index = min(frames_total - 1, frame_index + 1)
                    paused = True
                elif key == ord("1"):
                    options["classes"]["bar"] = not options["classes"]["bar"]
                elif key == ord("2"):
                    options["classes"]["bench"] = not options["classes"]["bench"]
                elif key == ord("3"):
                    options["classes"]["person"] = not options["classes"]["person"]
                elif key == ord("m"):
                    options["masks"] = not options["masks"]
                elif key == ord("b"):
                    options["boxes"] = not options["boxes"]
                elif key == ord("c"):
                    options["centroids"] = not options["centroids"]
                elif key == ord("t"):
                    options["tracks"] = not options["tracks"]
                elif key == ord("p"):
                    options["main"] = not options["main"]
                elif key == ord("f"):
                    options["confidence"] = not options["confidence"]
                elif key == ord("i"):
                    options["interactions"] = not options["interactions"]
                elif not paused:
                    frame_index = min(frames_total - 1, frame_index + 1)
                    if frame_index == frames_total - 1:
                        paused = True
        return 0
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
