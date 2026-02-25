import cv2
import numpy as np

# ----------------------------------
# CONFIG
# ----------------------------------
CAM = "left"   # "left" o "right"

video_path = f"../dataset/sq/sq_080/cam_{CAM}.mp4"
body_path  = f"../processed/sq/sq_080/{CAM}_body.npy"

# ----------------------------------
# CARGAR DATOS
# ----------------------------------
body = np.load(body_path)   # (T, 33, 3)

cap = cv2.VideoCapture(video_path)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_idx = 0

HIP_L = 23
HIP_R = 24

# ----------------------------------
# LOOP
# ----------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= len(body):
        break

    hl = body[frame_idx][HIP_L]
    hr = body[frame_idx][HIP_R]

    # -------------------------------
    # CADERA IZQ / DER
    # -------------------------------
    if hl[2] > 0.5:
        cv2.circle(
            frame,
            (int(hl[0] * w), int(hl[1] * h)),
            6,
            (0, 255, 0),
            -1
        )

    if hr[2] > 0.5:
        cv2.circle(
            frame,
            (int(hr[0] * w), int(hr[1] * h)),
            6,
            (0, 255, 0),
            -1
        )

    # -------------------------------
    # CENTRO DE CADERA
    # -------------------------------
    if hl[2] > 0.5 and hr[2] > 0.5:
        cx = int(((hl[0] + hr[0]) / 2) * w)
        cy = int(((hl[1] + hr[1]) / 2) * h)

        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

    # -------------------------------
    # INFO
    # -------------------------------
    cv2.putText(
        frame,
        f"{CAM.upper()} | HIP ONLY | Frame {frame_idx}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow(f"Hip {CAM}", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()