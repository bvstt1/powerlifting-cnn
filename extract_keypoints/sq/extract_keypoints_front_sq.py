import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


# -----------------------------------
# CONFIG
# -----------------------------------

MODEL_PATH = "../../models/pose_landmarker_heavy.task"

DATASET_ROOT = Path(
    r"C:\Users\basti\MediapipePythonProjects\dataset"
)

OUTPUT_ROOT = Path(
    r"C:\Users\basti\MediapipePythonProjects\keypoints"
)

# Keypoints faciales
FACE_IDXS = [0,1,2,3,4,5,6,7,8,9,10]


# -----------------------------------
# Crear landmarker
# -----------------------------------

def create_landmarker():

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=vision.RunningMode.VIDEO
    )

    return vision.PoseLandmarker.create_from_options(options)


# -----------------------------------
# Extraer keypoints
# -----------------------------------

def extract_keypoints_from_video(video_path):

    # NUEVO LANDMARKER POR VIDEO
    landmarker = create_landmarker()

    cap = cv2.VideoCapture(str(video_path))

    all_keypoints = []

    frame_idx = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # Timestamp estrictamente creciente
        timestamp_ms = frame_idx * 33

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # -----------------------------------
        # Pose detectada
        # -----------------------------------

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]

            frame_kps = []

            for idx, lm in enumerate(landmarks):

                # eliminar cara
                if idx in FACE_IDXS:
                    continue

                frame_kps.append([
                    lm.x,
                    lm.y,
                    lm.visibility
                ])

            frame_kps = np.array(
                frame_kps,
                dtype=np.float32
            )

        else:

            frame_kps = np.full(
                (22, 3),
                np.nan,
                dtype=np.float32
            )

        all_keypoints.append(frame_kps)

        frame_idx += 1

    cap.release()

    landmarker.close()

    return np.array(
        all_keypoints,
        dtype=np.float32
    )


# -----------------------------------
# Procesar squat frontal
# -----------------------------------

def process_squat_front():

    input_dir = DATASET_ROOT / "sq" / "front"

    output_dir = OUTPUT_ROOT / "sq" / "front"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    video_files = sorted(
        input_dir.glob("*.mp4")
    )

    print(f"\nVideos encontrados: {len(video_files)}")

    for video_path in video_files:

        print(f"\n→ Procesando {video_path.name}")

        keypoints = extract_keypoints_from_video(
            video_path
        )

        out_file = (
            output_dir
            / f"{video_path.stem}.npy"
        )

        np.save(out_file, keypoints)

        print(f"✔ Guardado: {out_file}")
        print(f"✔ Shape: {keypoints.shape}")

    print("\nDataset procesado correctamente")


# -----------------------------------
# MAIN
# -----------------------------------

if __name__ == "__main__":

    process_squat_front()