import cv2
import numpy as np

# ----------------------------------
# CONFIG
# ----------------------------------
CAM = "right"   # "left" o "right"

video_path = f"../../dataset/sq/sq_040/cam_{CAM}.mp4"
body_path  = f"../../processed/sq/sq_040/{CAM}_body.npy"
output_path = f"../../visualizations/sq_040_{CAM}_leg_only.mp4"

# ----------------------------------
# CARGAR DATOS
# ----------------------------------
body = np.load(body_path)   # (T, 33, 3)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

frame_idx = 0

# ----------------------------------
# KEYPOINTS SEGÚN VISTA
# ----------------------------------
if CAM == "right":
    HIP = 24
    KNEE = 26
    ANKLE = 28
    FOOT = 32
else:
    HIP = 23
    KNEE = 25
    ANKLE = 27
    FOOT = 31

LEG_CONNECTIONS = [
    (HIP, KNEE),
    (KNEE, ANKLE),
    (ANKLE, FOOT),
]

# ----------------------------------
# LOOP
# ----------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= len(body):
        break

    # -------------------------------
    # DIBUJAR PUNTOS
    # -------------------------------
    for idx in [HIP, KNEE, ANKLE, FOOT]:
        x, y, v = body[frame_idx][idx]
        if v > 0.5:
            cx = int(x * w)
            cy = int(y * h)
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

    # -------------------------------
    # DIBUJAR ESQUELETO
    # -------------------------------
    for a, b in LEG_CONNECTIONS:
        xa, ya, va = body[frame_idx][a]
        xb, yb, vb = body[frame_idx][b]

        if va > 0.5 and vb > 0.5:
            pa = (int(xa * w), int(ya * h))
            pb = (int(xb * w), int(yb * h))
            cv2.line(frame, pa, pb, (255, 0, 0), 3)

    # -------------------------------
    # INFO
    # -------------------------------
    cv2.putText(
        frame,
        f"{CAM.upper()} | LEG ONLY | Frame {frame_idx}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    out.write(frame)
    cv2.imshow("Leg only", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video guardado en: {output_path}")