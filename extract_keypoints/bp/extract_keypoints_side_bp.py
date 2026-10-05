from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# -----------------------------------
# CONFIG
# -----------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "bp" / "bp_side_skeleton_v2.pt"
DATASET_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_ROOT = PROJECT_ROOT / "keypoints"

SIDE_VIEWS = ("left", "right")
NUM_KEYPOINTS = 8
KEYPOINT_DIMS = 3
IMG_SIZE = 960
CONF_THRESHOLD = 0.5
BATCH_SIZE = 16
DEVICE = 0 if torch.cuda.is_available() else "cpu"


# -----------------------------------
# Extraer keypoints
# -----------------------------------

def extract_keypoints_from_result(result):
    frame_kps = np.full(
        (NUM_KEYPOINTS, KEYPOINT_DIMS),
        np.nan,
        dtype=np.float32
    )

    if result.keypoints is None or len(result.keypoints) == 0:
        return frame_kps

    if result.boxes is not None and len(result.boxes) > 0:
        person_idx = int(np.argmax(result.boxes.conf.cpu().numpy()))
    else:
        person_idx = 0

    coordinates = (
        result.keypoints.xyn[person_idx]
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if result.keypoints.conf is not None:
        confidence = (
            result.keypoints.conf[person_idx]
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    else:
        confidence = np.ones(NUM_KEYPOINTS, dtype=np.float32)

    if coordinates.shape != (NUM_KEYPOINTS, 2):
        raise RuntimeError(
            "El modelo lateral de press de banca debe entregar "
            f"{NUM_KEYPOINTS} keypoints, recibio {coordinates.shape}"
        )

    valid = (
        (confidence >= CONF_THRESHOLD)
        & np.any(coordinates != 0, axis=1)
    )

    frame_kps[:, 2] = confidence
    frame_kps[valid, :2] = coordinates[valid]

    return frame_kps


def extract_keypoints_from_video(video_path, model):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    all_keypoints = []

    try:
        while True:
            frames = []

            for _ in range(BATCH_SIZE):
                ret, frame = cap.read()

                if not ret:
                    break

                frames.append(frame)

            if not frames:
                break

            results = model(
                frames,
                imgsz=IMG_SIZE,
                device=DEVICE,
                verbose=False
            )

            all_keypoints.extend(
                extract_keypoints_from_result(result)
                for result in results
            )
    finally:
        cap.release()

    if not all_keypoints:
        return np.empty(
            (0, NUM_KEYPOINTS, KEYPOINT_DIMS),
            dtype=np.float32
        )

    return np.stack(all_keypoints).astype(np.float32)


# -----------------------------------
# Procesar press de banca lateral
# -----------------------------------

def process_bp_side_view(view, model):
    input_dir = DATASET_ROOT / "bp" / view
    output_dir = OUTPUT_ROOT / "bp" / view

    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(input_dir.glob("*.mp4"))

    if not video_files:
        raise RuntimeError(f"No se encontraron videos en: {input_dir}")

    print(f"\nVista {view} | Videos encontrados: {len(video_files)}")

    for index, video_path in enumerate(video_files, 1):
        out_file = output_dir / f"{video_path.stem}.npy"

        print(f"\n[{index}/{len(video_files)}] Procesando {video_path.name}")

        keypoints = extract_keypoints_from_video(video_path, model)
        np.save(out_file, keypoints)

        print(f"[OK] Guardado: {out_file}")
        print(f"[OK] Shape: {keypoints.shape}")

    print(f"\nDataset BP {view} procesado correctamente con YOLO")


def process_bp_side():
    model = YOLO(str(MODEL_PATH))
    kpt_shape = model.model.yaml.get("kpt_shape", [])

    if kpt_shape != [NUM_KEYPOINTS, KEYPOINT_DIMS]:
        raise RuntimeError(
            "Se esperaba un modelo de "
            f"{NUM_KEYPOINTS} keypoints x {KEYPOINT_DIMS} dimensiones: "
            f"{kpt_shape}"
        )

    print(f"Dispositivo: {DEVICE} | Batch: {BATCH_SIZE}")

    for view in SIDE_VIEWS:
        process_bp_side_view(view, model)


if __name__ == "__main__":
    process_bp_side()
