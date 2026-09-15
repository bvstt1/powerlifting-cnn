import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO


# -----------------------------------
# CONFIG
# -----------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "dl" / "dl_side_skeleton_v1.pt"
DATASET_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_ROOT = PROJECT_ROOT / "keypoints"

NUM_KEYPOINTS = 2  # hombro + cadera
IMG_SIZE = 960
CONF_THRESHOLD = 0.5
BATCH_SIZE = 16
DEVICE = 0 if torch.cuda.is_available() else "cpu"


# -----------------------------------
# Extraer keypoints
# -----------------------------------

def extract_keypoints_from_result(result):
    frame_kps = np.full(
        (NUM_KEYPOINTS, 3),
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
            "El modelo lateral debe entregar "
            f"{NUM_KEYPOINTS} keypoints, recibió {coordinates.shape}"
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

    cap.release()

    if not all_keypoints:
        return np.empty(
            (0, NUM_KEYPOINTS, 3),
            dtype=np.float32
        )

    return np.stack(all_keypoints).astype(np.float32)


# -----------------------------------
# Procesar deadlift izquierda
# -----------------------------------

def process_dl_left():
    input_dir = DATASET_ROOT / "dl" / "left"
    output_dir = OUTPUT_ROOT / "dl" / "left"

    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(input_dir.glob("*.mp4"))

    if not video_files:
        raise RuntimeError(f"No se encontraron videos en: {input_dir}")

    model = YOLO(str(MODEL_PATH))
    kpt_shape = model.model.yaml.get("kpt_shape", [])

    if not kpt_shape or kpt_shape[0] != NUM_KEYPOINTS:
        raise RuntimeError(
            f"Se esperaba un modelo de {NUM_KEYPOINTS} keypoints: {kpt_shape}"
        )

    print(f"\nVideos encontrados: {len(video_files)}")
    print(f"Dispositivo: {DEVICE} | Batch: {BATCH_SIZE}")

    for index, video_path in enumerate(video_files, 1):
        out_file = output_dir / f"{video_path.stem}.npy"

        if out_file.exists():
            print(f"[SKIP {index}/{len(video_files)}] {video_path.name}")
            continue

        print(f"\n[{index}/{len(video_files)}] Procesando {video_path.name}")

        keypoints = extract_keypoints_from_video(video_path, model)
        np.save(out_file, keypoints)

        print(f"[OK] Guardado: {out_file}")
        print(f"[OK] Shape: {keypoints.shape}")

    print("\nDataset DL left procesado correctamente con YOLO")


if __name__ == "__main__":
    process_dl_left()
