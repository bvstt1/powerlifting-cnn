import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# ----------------------------------
# CONFIG
# ----------------------------------
VIDEO_PATH = r"C:\Users\basti\MediapipePythonProjects\dataset\bp\left\bp_400.mp4"
MODEL_PATH = "../../models/pose_landmarker_heavy.task"

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 700

ALPHA = 0.25  # smoothing factor

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
# LOWER BODY CONNECTIONS
# ----------------------------------
connections = [
    (23,24),

    (23,25),
    (25,27),
    (27,29),
    (29,31),

    (24,26),
    (26,28),
    (28,30),
    (30,32)
]

# ----------------------------------
# LOWER BODY POINTS
# ----------------------------------
points = [
    23,24,
    25,26,
    27,28,
    29,30,
    31,32
]

# ----------------------------------
# EMA STORAGE
# ----------------------------------
smoothed_points = {}

# ----------------------------------
# VIDEO
# ----------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0

print("ESC para salir")

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    h, w, _ = frame.shape

    # ----------------------------------
    # MEDIAPIPE
    # ----------------------------------
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    timestamp_ms = int((frame_idx / max(fps, 1)) * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    # ----------------------------------
    # DRAW LOWER BODY
    # ----------------------------------
    if result.pose_landmarks:

        pose = result.pose_landmarks[0]

        # ----------------------------------
        # EMA SMOOTHING
        # ----------------------------------
        current_points = {}

        for idx in points:

            point = pose[idx]

            x = point.x * w
            y = point.y * h

            # primera vez
            if idx not in smoothed_points:
                smoothed_points[idx] = (x, y)

            prev_x, prev_y = smoothed_points[idx]

            # EMA
            smooth_x = ALPHA * x + (1 - ALPHA) * prev_x
            smooth_y = ALPHA * y + (1 - ALPHA) * prev_y

            smoothed_points[idx] = (smooth_x, smooth_y)

            current_points[idx] = (
                int(smooth_x),
                int(smooth_y)
            )

        # ----------------------------------
        # líneas
        # ----------------------------------
        for start_idx, end_idx in connections:

            x1, y1 = current_points[start_idx]
            x2, y2 = current_points[end_idx]

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (255,0,0),
                3
            )

        # ----------------------------------
        # puntos
        # ----------------------------------
        for idx in points:

            x, y = current_points[idx]

            cv2.circle(frame, (x, y), 10, (0,255,0), -1)
            cv2.circle(frame, (x, y), 4, (255,255,255), -1)

    frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    cv2.imshow("Lower Body Skeleton EMA", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
landmarker.close()
cv2.destroyAllWindows()