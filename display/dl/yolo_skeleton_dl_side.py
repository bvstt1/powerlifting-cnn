from ultralytics import YOLO
import cv2
import numpy as np

# -------------------------
# CONFIG
# -------------------------

model = YOLO("../../models/dl_side_skeleton_v1.pt")

video_path = r"C:\Users\basti\MediapipePythonProjects\dataset\dl\right\dl_403.mp4"

# Suavizado EMA
alpha = 0.6

# Conexiones skeleton (un solo lado - lateral)
# El modelo detecta los keypoints visibles desde la vista lateral
SKELETON = [
    (0, 1),  # hombro -> cadera
    (1, 2),  # cadera -> rodilla
    (2, 3),  # rodilla -> tobillo
    (3, 4),  # tobillo -> pie
]

# -------------------------
# VIDEO
# -------------------------

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error al abrir video")
    exit()

window_name = "DL Side - YOLO Skeleton"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 900, 600)

prev_kpts = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)

    annotated = frame.copy()

    if len(results) > 0 and results[0].keypoints is not None:

        kpts = results[0].keypoints.xy.cpu().numpy()

        if len(kpts) > 0:

            person = kpts[0]

            # Suavizado EMA
            if prev_kpts is not None:
                person = alpha * person + (1 - alpha) * prev_kpts

            prev_kpts = person.copy()

            # Dibujar lineas
            for p1, p2 in SKELETON:

                if p1 >= len(person) or p2 >= len(person):
                    continue

                x1, y1 = person[p1]
                x2, y2 = person[p2]

                if x1 == 0 and y1 == 0:
                    continue
                if x2 == 0 and y2 == 0:
                    continue

                cv2.line(
                    annotated,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255, 0, 0),
                    3
                )

            # Dibujar puntos
            for idx, (x, y) in enumerate(person):

                if x == 0 and y == 0:
                    continue

                cv2.circle(
                    annotated,
                    (int(x), int(y)),
                    6,
                    (0, 255, 0),
                    -1
                )

    cv2.imshow(window_name, annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
