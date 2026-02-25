import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# ----------------------------------
# CONFIG
# ----------------------------------
MODEL_PATH = "../../models/pose_landmarker_heavy.task"
CONF_THRESHOLD = 0.5

# ----------------------------------
# LANDMARKER
# ----------------------------------
def create_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO
    )
    return vision.PoseLandmarker.create_from_options(options)

landmarker = create_landmarker()

# ----------------------------------
# CONEXIONES DEL ESQUELETO
# ----------------------------------
POSE_CONNECTIONS = [
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 12),
    (11, 23), (12, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

# ----------------------------------
# WEBCAM
# ----------------------------------
cap = cv2.VideoCapture(1)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0

print("Presiona ESC para salir")

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

    timestamp_ms = int((frame_idx / max(fps, 1)) * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]

        # -------------------------------
        # DIBUJAR PUNTOS
        # -------------------------------
        for lm in landmarks:
            if lm.visibility < CONF_THRESHOLD:
                continue
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # -------------------------------
        # DIBUJAR ESQUELETO
        # -------------------------------
        for a, b in POSE_CONNECTIONS:
            la = landmarks[a]
            lb = landmarks[b]

            if la.visibility < CONF_THRESHOLD or lb.visibility < CONF_THRESHOLD:
                continue

            pa = (int(la.x * w), int(la.y * h))
            pb = (int(lb.x * w), int(lb.y * h))
            cv2.line(frame, pa, pb, (255, 0, 0), 2)

    # -------------------------------
    # INFO
    # -------------------------------
    cv2.putText(
        frame,
        "MediaPipe LIVE - Ajusta camara",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("Live Pose Check", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
landmarker.close()
cv2.destroyAllWindows()