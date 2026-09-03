import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# ----------------------------------
# CONFIG
# ----------------------------------

VIDEO_PATH = r"C:\Users\basti\MediapipePythonProjects\dataset\dl\left\dl_004.mp4"
MODEL_PATH = "../../models/pose_landmarker_heavy.task"

CONF_THRESHOLD = 0.3
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 720

# Landmarks de MediaPipe
# 11 = hombro izquierdo, 12 = hombro derecho
# 23 = cadera izquierda, 24 = cadera derecha

SIDES = {
    "left":  (11, 23),
    "right": (12, 24),
}


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

        # Elegir el lado mas visible
        side_name = None
        side_score = 0.0

        for name, (shoulder, hip) in SIDES.items():

            vis = min(landmarks[shoulder].visibility, landmarks[hip].visibility)

            if vis >= CONF_THRESHOLD and vis > side_score:
                side_score = vis
                side_name = name

        if side_name is not None:

            shoulder, hip = SIDES[side_name]

            # Dibujar puntos
            for idx in (shoulder, hip):

                lm = landmarks[idx]

                x = int(lm.x * w)
                y = int(lm.y * h)

                cv2.circle(frame, (x,y), 5, (0,255,0), -1)

            # Dibujar conexion hombro-cadera
            la = landmarks[shoulder]
            lb = landmarks[hip]

            x1 = int(la.x * w)
            y1 = int(la.y * h)

            x2 = int(lb.x * w)
            y2 = int(lb.y * h)

            cv2.line(frame, (x1,y1), (x2,y2), (255,0,0), 3)

            cv2.putText(frame, f"LADO: {side_name.upper()}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

    frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    cv2.imshow("DL Side - MediaPipe Skeleton", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_idx += 1


cap.release()
landmarker.close()
cv2.destroyAllWindows()
