import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import time
import numpy as np
import math

# ----------------------------------
# CONFIG
# ----------------------------------
VIDEO_PATH = "../../dataset/bp/bp_002/cam_left.mp4"
MODEL_PATH = "../../models/pose_landmarker_heavy.task"

CONF_THRESHOLD = 0.5
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 700
GRAPH_HEIGHT = 200

ALPHA = 0.2

USE_RIGHT_LEG = False

if USE_RIGHT_LEG:
    HIP_ID = 24
    KNEE_ID = 26
    ANKLE_ID = 28
else:
    HIP_ID = 23
    KNEE_ID = 25
    ANKLE_ID = 27

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
# SUAVIZADO
# ----------------------------------
prev_points = {}

def smooth_point(key, new_point):
    if key not in prev_points:
        prev_points[key] = new_point
        return new_point

    prev = prev_points[key]
    smoothed = (
        ALPHA * new_point[0] + (1 - ALPHA) * prev[0],
        ALPHA * new_point[1] + (1 - ALPHA) * prev[1],
    )
    prev_points[key] = smoothed
    return smoothed

# ----------------------------------
# ANGULO RODILLA
# ----------------------------------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

# ----------------------------------
# VIDEO
# ----------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0

hip_y_vals = []
knee_y_vals = []
ankle_y_vals = []

MAX_POINTS = 300

print("ESC para salir")

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

    graph = np.zeros((GRAPH_HEIGHT, w, 3), dtype=np.uint8)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]

        hip = landmarks[HIP_ID]
        knee = landmarks[KNEE_ID]
        ankle = landmarks[ANKLE_ID]

        if (
            hip.visibility > CONF_THRESHOLD and
            knee.visibility > CONF_THRESHOLD and
            ankle.visibility > CONF_THRESHOLD
        ):
            hip_pt = (hip.x * w, hip.y * h)
            knee_pt = (knee.x * w, knee.y * h)
            ankle_pt = (ankle.x * w, ankle.y * h)

            hip_pt = smooth_point("hip", hip_pt)
            knee_pt = smooth_point("knee", knee_pt)
            ankle_pt = smooth_point("ankle", ankle_pt)

            # Dibujar esqueleto
            cv2.circle(frame, (int(hip_pt[0]), int(hip_pt[1])), 6, (0,255,0), -1)
            cv2.circle(frame, (int(knee_pt[0]), int(knee_pt[1])), 6, (0,255,0), -1)
            cv2.circle(frame, (int(ankle_pt[0]), int(ankle_pt[1])), 6, (0,0,255), -1)

            cv2.line(frame,
                     (int(hip_pt[0]), int(hip_pt[1])),
                     (int(knee_pt[0]), int(knee_pt[1])),
                     (255,0,0), 3)

            cv2.line(frame,
                     (int(knee_pt[0]), int(knee_pt[1])),
                     (int(ankle_pt[0]), int(ankle_pt[1])),
                     (255,0,0), 3)

            # Guardar valores Y (invertimos para gráfico natural)
            hip_y_vals.append(h - hip_pt[1])
            knee_y_vals.append(h - knee_pt[1])
            ankle_y_vals.append(h - ankle_pt[1])

            if len(hip_y_vals) > MAX_POINTS:
                hip_y_vals.pop(0)
                knee_y_vals.pop(0)
                ankle_y_vals.pop(0)

    # ----------------------------------
    # GRAFICAR
    # ----------------------------------
    for i in range(1, len(hip_y_vals)):
        x1 = int((i-1) * w / MAX_POINTS)
        x2 = int(i * w / MAX_POINTS)

        y1 = int(GRAPH_HEIGHT - hip_y_vals[i-1] / h * GRAPH_HEIGHT)
        y2 = int(GRAPH_HEIGHT - hip_y_vals[i] / h * GRAPH_HEIGHT)
        cv2.line(graph, (x1,y1), (x2,y2), (0,255,0), 2)

        y1 = int(GRAPH_HEIGHT - knee_y_vals[i-1] / h * GRAPH_HEIGHT)
        y2 = int(GRAPH_HEIGHT - knee_y_vals[i] / h * GRAPH_HEIGHT)
        cv2.line(graph, (x1,y1), (x2,y2), (255,0,0), 2)

        y1 = int(GRAPH_HEIGHT - ankle_y_vals[i-1] / h * GRAPH_HEIGHT)
        y2 = int(GRAPH_HEIGHT - ankle_y_vals[i] / h * GRAPH_HEIGHT)
        cv2.line(graph, (x1,y1), (x2,y2), (0,0,255), 2)

    combined = np.vstack((frame, graph))
    combined = cv2.resize(combined, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    cv2.imshow("Lateral Leg + Motion Graph", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
landmarker.close()
cv2.destroyAllWindows()