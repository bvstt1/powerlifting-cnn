import cv2
import numpy as np
from pathlib import Path
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


# Configuración PoseLandmarker
MODEL_PATH = "models/pose_landmarker_heavy.task"
NUM_KEYPOINTS = 33


def create_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO
    )
    return vision.PoseLandmarker.create_from_options(options)


# Extraer keypoints de un video
def extract_keypoints_from_video(video_path, landmarker):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    all_keypoints = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = vision.MPImage(
            image_format=vision.ImageFormat.SRGB,
            data=frame_rgb
        )

        timestamp_ms = int((frame_idx / fps) * 1000)

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            frame_kps = np.array([
                [lm.x, lm.y, lm.visibility]
                for lm in landmarks
            ])
        else:
            frame_kps = np.full((NUM_KEYPOINTS, 3), np.nan)

        all_keypoints.append(frame_kps)
        frame_idx += 1

    cap.release()
    return np.array(all_keypoints)  # (T, 33, 3)


# Procesar un intento (3 cámaras)
def process_attempt(attempt_dir, output_dir, landmarker):
    cameras = {
        "front": "cam_front.mp4",
        "left": "cam_left.mp4",
        "right": "cam_right.mp4"
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    for cam_name, cam_file in cameras.items():
        video_path = attempt_dir / cam_file
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        keypoints = extract_keypoints_from_video(video_path, landmarker)

        out_file = output_dir / f"{attempt_dir.name}_{cam_name}.npy"
        np.save(out_file, keypoints)

        print(f"✔ {out_file} {keypoints.shape}")


#Procesar todo el dataset
def process_dataset(dataset_root, output_root):
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    landmarker = create_landmarker()

    for exercise_dir in dataset_root.iterdir():
        if not exercise_dir.is_dir():
            continue

        print(f"\n=== {exercise_dir.name.upper()} ===")

        for attempt_dir in exercise_dir.iterdir():
            if not attempt_dir.is_dir():
                continue

            print(f"→ Procesando {attempt_dir.name}")
            out_dir = output_root / exercise_dir.name / attempt_dir.name
            process_attempt(attempt_dir, out_dir, landmarker)

    landmarker.close()


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    process_dataset(
        dataset_root="dataset",
        output_root="keypoints"
    )
