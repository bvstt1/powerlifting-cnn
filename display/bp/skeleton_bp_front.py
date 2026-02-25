import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import numpy as np
import time

# ----------------------------------
# CONFIG
# ----------------------------------
VIDEO_PATH = "../../dataset/bp/bp_002/cam_front.mp4"
MODEL_PATH = "../../models/pose_landmarker_heavy.task"

CONF_THRESHOLD = 0.5
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 720
GRAPH_HEIGHT = 200
MAX_POINTS = 300
ALPHA = 0.1

# Landmarks torso + brazos
UPPER_BODY_LANDMARKS = [11,12,13,14,15,16,23,24]

UPPER_BODY_CONNECTIONS = [
    (11,12),
    (23,24),
    (11,23),
    (12,24),
    (11,13),(13,15),
    (12,14),(14,16),
]

LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

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
# SUAVIZADO EMA
# ----------------------------------
prev_points = {}

def smooth_point(key, point):
    if key not in prev_points:
        prev_points[key] = point
        return point

    prev = prev_points[key]
    smoothed = (
        ALPHA * point[0] + (1 - ALPHA) * prev[0],
        ALPHA * point[1] + (1 - ALPHA) * prev[1]
    )
    prev_points[key] = smoothed
    return smoothed

# ----------------------------------
# VIDEO
# ----------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0

left_wrist_vals = []
right_wrist_vals = []
shoulder_tilt_vals = []

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

    timestamp_ms = int((frame_idx / max(fps,1)) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    graph = np.zeros((GRAPH_HEIGHT, w, 3), dtype=np.uint8)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]

        # Dibujar cuerpo superior
        for idx in UPPER_BODY_LANDMARKS:
            lm = landmarks[idx]
            if lm.visibility < CONF_THRESHOLD:
                continue
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            cv2.circle(frame,(cx,cy),5,(0,255,0),-1)

        for a,b in UPPER_BODY_CONNECTIONS:
            la = landmarks[a]
            lb = landmarks[b]
            if la.visibility < CONF_THRESHOLD or lb.visibility < CONF_THRESHOLD:
                continue
            pa = (int(la.x*w), int(la.y*h))
            pb = (int(lb.x*w), int(lb.y*h))
            cv2.line(frame,pa,pb,(255,0,0),3)

        # --- Extraer y suavizar puntos clave ---
        lw = landmarks[LEFT_WRIST]
        rw = landmarks[RIGHT_WRIST]
        ls = landmarks[LEFT_SHOULDER]
        rs = landmarks[RIGHT_SHOULDER]

        if (lw.visibility>CONF_THRESHOLD and
            rw.visibility>CONF_THRESHOLD and
            ls.visibility>CONF_THRESHOLD and
            rs.visibility>CONF_THRESHOLD):

            lw_pt = smooth_point("lw",(lw.x*w,lw.y*h))
            rw_pt = smooth_point("rw",(rw.x*w,rw.y*h))
            ls_pt = smooth_point("ls",(ls.x*w,ls.y*h))
            rs_pt = smooth_point("rs",(rs.x*w,rs.y*h))

            # Altura invertida (para gráfico natural)
            left_wrist_vals.append(h - lw_pt[1])
            right_wrist_vals.append(h - rw_pt[1])
            shoulder_tilt_vals.append(ls_pt[1] - rs_pt[1])

            if len(left_wrist_vals) > MAX_POINTS:
                left_wrist_vals.pop(0)
                right_wrist_vals.pop(0)
                shoulder_tilt_vals.pop(0)

    # ----------------------------------
    # GRAFICO
    # ----------------------------------
    for i in range(1,len(left_wrist_vals)):
        x1 = int((i-1)*w/MAX_POINTS)
        x2 = int(i*w/MAX_POINTS)

        # Muñeca izquierda (verde)
        y1 = int(GRAPH_HEIGHT - left_wrist_vals[i-1]/h*GRAPH_HEIGHT)
        y2 = int(GRAPH_HEIGHT - left_wrist_vals[i]/h*GRAPH_HEIGHT)
        cv2.line(graph,(x1,y1),(x2,y2),(0,255,0),2)

        # Muñeca derecha (azul)
        y1 = int(GRAPH_HEIGHT - right_wrist_vals[i-1]/h*GRAPH_HEIGHT)
        y2 = int(GRAPH_HEIGHT - right_wrist_vals[i]/h*GRAPH_HEIGHT)
        cv2.line(graph,(x1,y1),(x2,y2),(255,0,0),2)

        # Inclinación hombros (rojo)
        tilt_scaled_prev = shoulder_tilt_vals[i-1]*2 + GRAPH_HEIGHT//2
        tilt_scaled_curr = shoulder_tilt_vals[i]*2 + GRAPH_HEIGHT//2
        cv2.line(graph,
                 (x1,int(tilt_scaled_prev)),
                 (x2,int(tilt_scaled_curr)),
                 (0,0,255),2)

    combined = np.vstack((frame, graph))
    combined = cv2.resize(combined,(DISPLAY_WIDTH,DISPLAY_HEIGHT))

    cv2.imshow("Upper Body + Motion Graph", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
landmarker.close()
cv2.destroyAllWindows()