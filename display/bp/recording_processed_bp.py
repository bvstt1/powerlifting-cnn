import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import numpy as np
import os

# ----------------------------------
# CONFIG
# ----------------------------------
BASE_PATH = "../../dataset/bp/bp_001"

VIDEO_FRONT = f"{BASE_PATH}/cam_front.mp4"
VIDEO_LEFT  = f"{BASE_PATH}/cam_left.mp4"
VIDEO_RIGHT = f"{BASE_PATH}/cam_right.mp4"

MODEL_PATH = "../../models/pose_landmarker_heavy.task"

OUTPUT_BASE = "../processed/bp"

CONF_THRESHOLD = 0.6

# ----------------------------------
# FUNCION: Obtener siguiente numero bp_xxx
# ----------------------------------
def get_next_attempt_number(base_path):
    if not os.path.exists(base_path):
        return 1

    existing = [d for d in os.listdir(base_path) if d.startswith("bp_")]
    if not existing:
        return 1

    nums = []
    for d in existing:
        try:
            nums.append(int(d.split("_")[1]))
        except:
            pass

    if not nums:
        return 1

    return max(nums) + 1

# ----------------------------------
# LANDMARKER
# ----------------------------------
def create_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO
    )
    return vision.PoseLandmarker.create_from_options(options)

landmarker_front = create_landmarker()
landmarker_left  = create_landmarker()
landmarker_right = create_landmarker()

landmarkers = [landmarker_front, landmarker_left, landmarker_right]

# ----------------------------------
# VIDEO CAPTURES
# ----------------------------------
cap_front = cv2.VideoCapture(VIDEO_FRONT)
cap_left  = cv2.VideoCapture(VIDEO_LEFT)
cap_right = cv2.VideoCapture(VIDEO_RIGHT)

fps = cap_front.get(cv2.CAP_PROP_FPS)
frame_idx = 0

recording = False

seq_front = []
seq_left = []
seq_right = []

attempt_counter = get_next_attempt_number(OUTPUT_BASE)

print("S = start lift")
print("E = end lift")
print("Q = quit")

# ----------------------------------
# LOOP
# ----------------------------------
while True:

    ret_f, frame_f = cap_front.read()
    ret_l, frame_l = cap_left.read()
    ret_r, frame_r = cap_right.read()

    if not (ret_f and ret_l and ret_r):
        break

    frames = [frame_f, frame_l, frame_r]
    processed_frames = []

    for i, frame in enumerate(frames):

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        timestamp_ms = int((frame_idx / max(fps, 1)) * 1000)
        result = landmarkers[i].detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            # Dibujar landmarks visibles
            for lm in landmarks:
                if lm.visibility > CONF_THRESHOLD:
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            # Guardar si estamos grabando
            if recording:
                frame_landmarks = []
                for lm in landmarks:
                    frame_landmarks.append([lm.x, lm.y, lm.z])

                if i == 0:
                    seq_front.append(frame_landmarks)
                elif i == 1:
                    seq_left.append(frame_landmarks)
                else:
                    seq_right.append(frame_landmarks)

        processed_frames.append(frame)

    # ----------------------------------
    # Unir vistas
    # ----------------------------------
    combined = np.hstack(processed_frames)

    if recording:
        cv2.putText(
            combined,
            "RECORDING",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

    cv2.imshow("BP Multi-View Segmentation", combined)

    key = cv2.waitKey(1) & 0xFF

    # ----------------------------------
    # START RECORDING
    # ----------------------------------
    if key == ord('s'):
        recording = True
        seq_front = []
        seq_left = []
        seq_right = []
        print("Started recording")

    # ----------------------------------
    # END RECORDING Y GUARDAR
    # ----------------------------------
    elif key == ord('e'):

        if recording:
            recording = False

            attempt_name = f"bp_{attempt_counter:03d}"
            attempt_dir = os.path.join(OUTPUT_BASE, attempt_name)
            os.makedirs(attempt_dir, exist_ok=True)

            np.save(os.path.join(attempt_dir, "front_body.npy"), np.array(seq_front))
            np.save(os.path.join(attempt_dir, "left_body.npy"), np.array(seq_left))
            np.save(os.path.join(attempt_dir, "right_body.npy"), np.array(seq_right))

            print(f"Saved {attempt_name}")
            print("Front shape:", np.array(seq_front).shape)

            attempt_counter += 1

    # ----------------------------------
    # QUIT
    # ----------------------------------
    elif key == ord('q'):
        break

    frame_idx += 1

# ----------------------------------
# CLEANUP
# ----------------------------------
cap_front.release()
cap_left.release()
cap_right.release()

for lm in landmarkers:
    lm.close()

cv2.destroyAllWindows()

print("Done.")