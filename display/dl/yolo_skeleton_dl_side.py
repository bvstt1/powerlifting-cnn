from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "dl" / "dl_side_skeleton_v1.pt"
VIDEO_PATH = PROJECT_ROOT / "dataset" / "dl" / "left" / "dl_408.mp4"

model = YOLO(str(MODEL_PATH))

# Suavizado EMA
alpha = 0.6
CONF_THRESHOLD = 0.5
IMG_SIZE = 960

# El modelo lateral detecta solamente hombro y cadera.
KEYPOINT_NAMES = ("cadera", "hombro")
SKELETON = [
    (1, 0),  # hombro -> cadera
]

# -------------------------
# VIDEO
# -------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise SystemExit(f"Error al abrir video: {VIDEO_PATH}")

window_name = "DL Side - YOLO Skeleton"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 900, 600)

prev_kpts = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, imgsz=IMG_SIZE, verbose=False)

    annotated = frame.copy()

    if results and results[0].keypoints is not None:

        kpts = results[0].keypoints.xy.cpu().numpy()
        kpts_conf = results[0].keypoints.conf

        if len(kpts) > 0:

            # Usar la persona con mayor confianza si hay varias detecciones.
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                person_idx = int(np.argmax(results[0].boxes.conf.cpu().numpy()))
            else:
                person_idx = 0

            person = kpts[person_idx]

            if len(person) != len(KEYPOINT_NAMES):
                raise RuntimeError(
                    f"Se esperaban 2 keypoints y el modelo entregó {len(person)}"
                )

            if kpts_conf is not None:
                confidence = kpts_conf[person_idx].cpu().numpy()
            else:
                confidence = np.ones(len(person), dtype=np.float32)

            valid = (
                (confidence >= CONF_THRESHOLD)
                & np.any(person != 0, axis=1)
            )

            # Suavizar solamente detecciones válidas para evitar puntos fantasma.
            if prev_kpts is None:
                prev_kpts = np.full_like(person, np.nan)

            for idx in range(len(person)):
                if not valid[idx]:
                    prev_kpts[idx] = np.nan
                    continue

                if np.all(np.isfinite(prev_kpts[idx])):
                    person[idx] = (
                        alpha * person[idx]
                        + (1 - alpha) * prev_kpts[idx]
                    )

                prev_kpts[idx] = person[idx]

            # Dibujar lineas
            for p1, p2 in SKELETON:

                if not valid[p1] or not valid[p2]:
                    continue

                x1, y1 = person[p1]
                x2, y2 = person[p2]

                cv2.line(
                    annotated,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255, 0, 0),
                    3
                )

            # Dibujar hombro y cadera con su confianza.
            for idx, (x, y) in enumerate(person):

                if not valid[idx]:
                    continue

                point = (int(x), int(y))

                cv2.circle(
                    annotated,
                    point,
                    6,
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    annotated,
                    f"{KEYPOINT_NAMES[idx]} {confidence[idx]:.2f}",
                    (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

        else:
            prev_kpts = None
    else:
        prev_kpts = None

    cv2.imshow(window_name, annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
