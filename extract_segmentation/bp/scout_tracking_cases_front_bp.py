import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "bp" / "bp_front_seg_v2.pt"
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "bp" / "front"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "segmentation" / "bp" / "front" / "manifests"
    / "bp_front_tracking_scout.csv"
)
EXPECTED_CLASSES = {0: "bar", 1: "bench", 2: "person"}
FIELDS = [
    "video_id", "frames_video", "sampled_frames", "samples",
    "bar_count_min", "bar_count_max", "bar_missing_samples",
    "bar_multiple_samples", "bar_confidence_min", "bar_confidence_max",
    "bar_confidence_range", "bench_count_min", "bench_count_max",
    "bench_missing_samples", "bench_multiple_samples", "bench_confidence_min",
    "bench_confidence_max", "bench_confidence_range", "person_count_min",
    "person_count_max", "person_missing_samples", "person_multiple_samples",
    "person_confidence_min", "person_confidence_max", "person_confidence_range",
    "interest_score", "selection_reasons",
]


def atomic_write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def sample_indices(frames_total, samples):
    if frames_total <= 0:
        raise ValueError("Video sin frames declarados")
    count = min(samples, frames_total)
    return np.unique(np.linspace(0, frames_total - 1, count).round().astype(int))


def read_samples(video_path, samples):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError("No se pudo abrir %s" % video_path)
    frames_total = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    records = []
    try:
        for frame_index in sample_indices(frames_total, samples):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                raise OSError(
                    "No se pudo leer %s frame %d" % (video_path.name, frame_index)
                )
            records.append((video_path.stem, frames_total, int(frame_index), frame))
    finally:
        capture.release()
    return records


def summarize_video(video_id, frames_total, records):
    row = {
        "video_id": video_id,
        "frames_video": frames_total,
        "sampled_frames": "|".join(str(record["frame_index"]) for record in records),
        "samples": len(records),
    }
    reasons = []
    score = 0.0
    for class_name in EXPECTED_CLASSES.values():
        counts = np.asarray([
            record["counts"][class_name] for record in records
        ], dtype=np.int32)
        confidences = np.asarray([
            confidence
            for record in records
            for confidence in record["confidences"][class_name]
        ], dtype=np.float64)
        missing = int(np.count_nonzero(counts == 0))
        multiple = int(np.count_nonzero(counts > 1))
        confidence_min = float(confidences.min()) if confidences.size else None
        confidence_max = float(confidences.max()) if confidences.size else None
        confidence_range = (
            confidence_max - confidence_min if confidences.size else None
        )
        row.update({
            class_name + "_count_min": int(counts.min()),
            class_name + "_count_max": int(counts.max()),
            class_name + "_missing_samples": missing,
            class_name + "_multiple_samples": multiple,
            class_name + "_confidence_min": confidence_min,
            class_name + "_confidence_max": confidence_max,
            class_name + "_confidence_range": confidence_range,
        })
        if missing:
            reasons.append("%s_missing=%d" % (class_name, missing))
        if multiple:
            reasons.append("%s_multiple=%d" % (class_name, multiple))
        if confidence_range is not None and confidence_range >= 0.30:
            reasons.append("%s_conf_range=%.3f" % (class_name, confidence_range))

        missing_weight = {"person": 8.0, "bar": 4.0, "bench": 4.0}[class_name]
        multiple_weight = {"person": 10.0, "bar": 5.0, "bench": 2.0}[class_name]
        score += missing_weight * missing
        score += multiple_weight * multiple
        if confidence_range is not None:
            score += confidence_range
    row["interest_score"] = float(score)
    row["selection_reasons"] = ";".join(reasons) or "stable_in_sample"
    return row


def flush_batch(model, batch, device, imgsz, conf, iou, by_video):
    if not batch:
        return
    frames = [record[3] for record in batch]
    results = model.predict(
        frames,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        retina_masks=False,
        device=device,
        verbose=False,
    )
    for source, result in zip(batch, results):
        video_id, frames_total, frame_index, _ = source
        class_ids = (
            result.boxes.cls.detach().cpu().numpy().astype(int)
            if result.boxes is not None else np.empty(0, dtype=int)
        )
        confidences = (
            result.boxes.conf.detach().cpu().numpy().astype(float)
            if result.boxes is not None else np.empty(0, dtype=float)
        )
        record = {
            "frame_index": frame_index,
            "counts": {},
            "confidences": {},
        }
        for class_id, class_name in EXPECTED_CLASSES.items():
            selected = confidences[class_ids == class_id]
            record["counts"][class_name] = int(len(selected))
            record["confidences"][class_name] = selected.tolist()
        by_video.setdefault(video_id, {
            "frames_total": frames_total,
            "records": [],
        })["records"].append(record)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exploración limitada sin labels para casos difíciles BP_FRONT"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-videos", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    numeric = [args.samples, args.batch_size, args.imgsz, args.conf, args.iou]
    if any(not math.isfinite(float(value)) for value in numeric):
        raise ValueError("Argumentos numéricos no finitos")
    if args.samples <= 0 or args.batch_size <= 0 or args.imgsz <= 0:
        raise ValueError("samples, batch-size e imgsz deben ser positivos")
    if not (0 <= args.conf <= 1 and 0 <= args.iou <= 1):
        raise ValueError("conf/iou fuera de rango")

    video_paths = sorted(args.input_dir.glob("*.mp4"))
    if args.max_videos is not None:
        video_paths = video_paths[:args.max_videos]
    if not video_paths:
        raise FileNotFoundError("No hay videos en %s" % args.input_dir)

    model = YOLO(str(args.model))
    class_names = {int(key): str(value) for key, value in model.names.items()}
    if class_names != EXPECTED_CLASSES:
        raise ValueError(
            "Clases inesperadas: esperadas %s, recibidas %s"
            % (EXPECTED_CLASSES, class_names)
        )

    by_video = {}
    batch = []
    for video_number, video_path in enumerate(video_paths, start=1):
        batch.extend(read_samples(video_path, args.samples))
        while len(batch) >= args.batch_size:
            flush_batch(
                model, batch[:args.batch_size], args.device, args.imgsz,
                args.conf, args.iou, by_video,
            )
            batch = batch[args.batch_size:]
        if video_number % 50 == 0:
            print("Explorados %d/%d videos" % (video_number, len(video_paths)))
    flush_batch(
        model, batch, args.device, args.imgsz, args.conf, args.iou, by_video
    )

    rows = []
    for video_id, data in by_video.items():
        records = sorted(data["records"], key=lambda record: record["frame_index"])
        rows.append(summarize_video(video_id, data["frames_total"], records))
    rows.sort(key=lambda row: row["video_id"])
    atomic_write_csv(args.output, rows)
    interesting = sorted(
        rows,
        key=lambda row: (-float(row["interest_score"]), row["video_id"]),
    )[:10]
    print("Exploración terminada: %d videos, %d frames muestreados" % (
        len(rows), sum(int(row["samples"]) for row in rows),
    ))
    for row in interesting:
        print("%s score=%.3f %s" % (
            row["video_id"], row["interest_score"], row["selection_reasons"],
        ))
    print("CSV: %s" % args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
