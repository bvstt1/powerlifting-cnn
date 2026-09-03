from ultralytics import YOLO
import cv2
import numpy as np

# ==========================
# CONFIG
# ==========================

MODEL_PATH = "../../models/sq_front_skeleton_v4.pt"
VIDEO_PATH = r"C:\Users\basti\MediapipePythonProjects\dataset\sq\front\sq_151.mp4"
TEST = r"C:\Users\basti\Desktop\todo\test_sq6.mp4"

# Suavizado EMA
alpha = 0.6

# ==========================
# ESQUELETO
# ==========================

SKELETON = [

    # Hombros
    (0, 12),
    (1, 13),
    (0, 1),

    # Brazos
    (12, 15),
    (13, 14),

    # Tronco
    (0, 3),
    (1, 2),

    # Caderas
    (2, 3),

    # Pierna izquierda
    (2, 5),
    (4, 9),
    (9, 10),
    (9, 11),
    (11, 10),

    # Pierna derecha
    (3, 4),
    (5, 6),
    (6, 8),
    (6, 7),
    (8, 7)

]

# ==========================
# LOAD MODEL
# ==========================

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

cv2.namedWindow("YOLO Skeleton", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Skeleton", 1000, 700)

prev_kpts = None

# ==========================
# LOOP
# ==========================

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)

    if len(results[0].keypoints) > 0:

        kpts = results[0].keypoints.xy.cpu().numpy()[0]

        # ==========================
        # SUAVIZADO EMA
        # ==========================

        if prev_kpts is None:
            prev_kpts = kpts.copy()

        kpts = alpha * kpts + (1 - alpha) * prev_kpts

        prev_kpts = kpts.copy()

        # ==========================
        # DIBUJAR LÍNEAS
        # ==========================

        for p1, p2 in SKELETON:

            if p1 >= len(kpts) or p2 >= len(kpts):
                continue

            x1, y1 = kpts[p1]
            x2, y2 = kpts[p2]

            cv2.line(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (255, 0, 0),
                2
            )

        # ==========================
        # DIBUJAR KEYPOINTS
        # ==========================

        for idx, (x, y) in enumerate(kpts):

            cv2.circle(
                frame,
                (int(x), int(y)),
                5,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                frame,
                str(idx),
                (int(x) + 8, int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )

    # ==========================
    # REDIMENSIONAR VISTA
    # ==========================

    small_frame = cv2.resize(
        frame,
        None,
        fx=0.5,
        fy=0.5
    )

    cv2.imshow("YOLO Skeleton", small_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()