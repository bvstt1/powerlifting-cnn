import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


# ----------------------------------
# CONFIG
# ----------------------------------

VIDEO_PATH = r"C:\Users\basti\MediapipePythonProjects\dataset\dl\front\dl_010.mp4"
MODEL_PATH = "../../models/pose_landmarker_heavy.task"

CONF_THRESHOLD = 0.5
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 720

OFFSET_Y = 19 # px hacia abajo para ajustar el centroide al centro de la barra


# ----------------------------------
# LANDMARKS (sin rostro 0-10)
# ----------------------------------

MANO_IZQ = [15, 17, 19, 21]   # muneca, meique, indice, pulgar
MANO_DER = [16, 18, 20, 22]
MANOS = MANO_IZQ + MANO_DER

BODY = list(range(11, 33))

CONEXIONES = [
    # Tronco
    (11,12),
    (11,23),
    (12,24),
    (23,24),

    # Brazo izquierdo
    (11,13),
    (13,15),
    (15,17),
    (15,19),
    (15,21),
    (17,19),

    # Brazo derecho
    (12,14),
    (14,16),
    (16,18),
    (16,20),
    (16,22),
    (18,20),

    # Pierna izquierda
    (23,25),
    (25,27),
    (27,29),
    (27,31),
    (29,31),

    # Pierna derecha
    (24,26),
    (26,28),
    (28,30),
    (28,32),
    (30,32)
]


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
# AUX
# ----------------------------------

def centro_mano(landmarks, indices, w, h, offset_y=0):
    pts = []
    for idx in indices:
        lm = landmarks[idx]
        if lm.visibility >= CONF_THRESHOLD:
            pts.append((int(lm.x * w), int(lm.y * h)))
    if not pts:
        return None
    cx = int(sum(p[0] for p in pts) / len(pts))
    cy = int(sum(p[1] for p in pts) / len(pts)) + offset_y
    return (cx, cy)


# ----------------------------------
# VIDEO
# ----------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0

print("ESC para salir | Barra simulada entre manos - DL Front")

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    timestamp_ms = int((frame_idx / max(fps, 1)) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.pose_landmarks:

        lm = result.pose_landmarks[0]

        # Landmarks del cuerpo
        for idx in BODY:
            if lm[idx].visibility < CONF_THRESHOLD:
                continue
            x = int(lm[idx].x * w)
            y = int(lm[idx].y * h)
            if idx in MANOS:
                cv2.circle(frame, (x, y), 9, (0, 255, 255), -1)
            else:
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        # Conexiones
        for a, b in CONEXIONES:
            if lm[a].visibility < CONF_THRESHOLD or lm[b].visibility < CONF_THRESHOLD:
                continue
            x1 = int(lm[a].x * w)
            y1 = int(lm[a].y * h)
            x2 = int(lm[b].x * w)
            y2 = int(lm[b].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Centroide de cada mano
        izq = centro_mano(lm, MANO_IZQ, w, h, OFFSET_Y)
        der = centro_mano(lm, MANO_DER, w, h, OFFSET_Y)

        if izq:
            cv2.circle(frame, izq, 14, (0, 255, 255), -1)
        if der:
            cv2.circle(frame, der, 14, (0, 255, 255), -1)

        # Barra simulada entre las manos
        if izq and der:
            cv2.line(frame, izq, der, (0, 255, 255), 5)

    frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    cv2.imshow("DL Front - Barra simulada entre manos", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_idx += 1


cap.release()
landmarker.close()
cv2.destroyAllWindows()
