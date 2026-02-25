import cv2
import numpy as np

# -------------------------------
# RUTAS
# -------------------------------
video_path = "../../dataset/sq/sq_002/cam_front.mp4"
body_path  = "../../processed/sq/sq_002/front_body.npy"
bar_path   = "../../processed/sq/sq_002/front_bar.npy"

# -------------------------------
# CARGAR DATOS
# -------------------------------
body = np.load(body_path)   # (T, 33, 3)
bar  = np.load(bar_path)    # (T, 3) o (T,)

cap = cv2.VideoCapture(video_path)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# señal vertical de la barra
bar_y = bar[:, 0] if bar.ndim > 1 else bar

# normalizar a rango [0, 1]
bar_y_norm = (bar_y - bar_y.min()) / (bar_y.max() - bar_y.min())

frame_idx = 0

# conexiones básicas del esqueleto
POSE_CONNECTIONS = [
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 12),
    (11, 23), (12, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= len(body):
        break

    # -------------------------------
    # DIBUJAR KEYPOINTS
    # -------------------------------
    for x, y, v in body[frame_idx]:
        if np.isnan(x) or v < 0.5:
            continue
        cx = int(x * w)
        cy = int(y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    # -------------------------------
    # DIBUJAR ESQUELETO
    # -------------------------------
    for a, b in POSE_CONNECTIONS:
        xa, ya, va = body[frame_idx][a]
        xb, yb, vb = body[frame_idx][b]

        if va < 0.5 or vb < 0.5:
            continue

        pa = (int(xa * w), int(ya * h))
        pb = (int(xb * w), int(yb * h))
        cv2.line(frame, pa, pb, (255, 0, 0), 2)

    # -------------------------------
    # BARRA (SOLO ALTURA)
    # -------------------------------
    y_bar = int(bar_y_norm[frame_idx] * h)

    cv2.line(
        frame,
        (0, y_bar),
        (w, y_bar),
        (0, 0, 255),
        2
    )

    # -------------------------------
    # INFO
    # -------------------------------
    cv2.putText(
        frame,
        f"Frame {frame_idx}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("Pose + Bar (simple)", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()