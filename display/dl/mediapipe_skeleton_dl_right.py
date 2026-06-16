import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# ----------------------------------
# CONFIG
# ----------------------------------

VIDEO_PATH = r"C:\Users\basti\MediapipePythonProjects\dataset\dl\right\dl_080.mp4"
MODEL_PATH = "../../models/pose_landmarker_heavy.task"

CONF_THRESHOLD = 0.5
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 720


# ----------------------------------
# FULL BODY LANDMARKS
# ----------------------------------

LANDMARKS = [
    11, 23
]

CONNECTIONS = [


    # Tronco
    (11,23),


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

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    timestamp_ms = int((frame_idx / max(fps,1)) * 1000)

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    # ----------------------------------
    # DRAW
    # ----------------------------------

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        # Dibujar puntos
        for idx in LANDMARKS:

            lm = landmarks[idx]

            if lm.visibility < CONF_THRESHOLD:
                continue

            x = int(lm.x * w)
            y = int(lm.y * h)

            cv2.circle(frame, (x,y), 5, (0,255,0), -1)


        # Dibujar conexiones
        for a,b in CONNECTIONS:

            la = landmarks[a]
            lb = landmarks[b]

            if la.visibility < CONF_THRESHOLD or lb.visibility < CONF_THRESHOLD:
                continue

            x1 = int(la.x * w)
            y1 = int(la.y * h)

            x2 = int(lb.x * w)
            y2 = int(lb.y * h)

            cv2.line(frame, (x1,y1), (x2,y2), (255,0,0), 3)


    frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    cv2.imshow("Front Bench Raw", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_idx += 1


cap.release()
landmarker.close()
cv2.destroyAllWindows()