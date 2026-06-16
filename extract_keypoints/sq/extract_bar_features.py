import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


MODEL_PATH = "models/pose_landmarker_heavy.task"


def create_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO
    )
    return vision.PoseLandmarker.create_from_options(options)


def create_tracker():
    return cv2.TrackerCSRT_create()


def extract_bar_features(video_path):
    landmarker = create_landmarker()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    tracker = None
    bar_y = []
    timestamps = []

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        timestamp_ms = int((frame_idx / fps) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Inicializar tracker usando muñecas
        if tracker is None and result.pose_landmarks:
            lm = result.pose_landmarks[0]
            lw, rw = lm[15], lm[16]

            if lw.visibility > 0.6 and rw.visibility > 0.6:
                cx = int(((lw.x + rw.x) / 2) * w)
                cy = int(((lw.y + rw.y) / 2) * h)

                box = int(0.15 * w)
                x = max(0, cx - box // 2)
                y = max(0, cy - box // 2)

                tracker = create_tracker()
                tracker.init(frame, (x, y, box, box))

                bar_y.append(cy)
                timestamps.append(timestamp_ms / 1000)
                frame_idx += 1
                continue

        # Tracking
        if tracker is not None:
            success, bbox = tracker.update(frame)
            if success:
                _, y, _, h_box = bbox
                bar_y.append(y + h_box / 2)
            else:
                bar_y.append(np.nan)
        else:
            bar_y.append(np.nan)

        timestamps.append(timestamp_ms / 1000)
        frame_idx += 1

    cap.release()
    landmarker.close()

    bar_y = np.array(bar_y)
    time = np.array(timestamps)

    # Interpolación
    valid = ~np.isnan(bar_y)
    bar_y = np.interp(np.arange(len(bar_y)), np.where(valid)[0], bar_y[valid])

    # Señales derivadas
    velocity = np.gradient(bar_y, time)
    acceleration = np.gradient(velocity, time)

    features = np.stack([bar_y, velocity, acceleration], axis=1)

    # Normalización
    features = (features - features.mean(axis=0)) / features.std(axis=0)

    return features  # (T, 3)
