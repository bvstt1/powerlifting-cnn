from ultralytics import YOLO
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# ----------------------------------
# CONFIG
# ----------------------------------

VIDEO_PATH = r"C:\Users\basti\MediapipePythonProjects\dataset\dl\front\dl_001.mp4"
MODEL_PATH = "../../models/pose_landmarker_heavy.task"
YOLO_SEG_PATH = "../../models/dl_front_seg_v1.pt"

CONF_THRESHOLD = 0.5
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 720

# ----------------------------------
# FULL BODY LANDMARKS
# ----------------------------------

LANDMARKS = list(range(33))

CONNECTIONS = [

    # Cara
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
    (9,10),

    # Tronco
    (11,12),
    (11,23),
    (12,24),
    (23,24),

    # Brazo izquierdo
    (11,13),
    (13,15),
    (15,17),
    (15,19),
    (15,21),
    (17,19),

    # Brazo derecho
    (12,14),
    (14,16),
    (16,18),
    (16,20),
    (16,22),
    (18,20),

    # Pierna izquierda
    (23,25),
    (25,27),
    (27,29),
    (27,31),
    (29,31),

    # Pierna derecha
    (24,26),
    (26,28),
    (28,30),
    (28,32),
    (30,32)
]


# ----------------------------------
# MODELOS
# ----------------------------------

def create_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO
    )
    return vision.PoseLandmarker.create_from_options(options)

landmarker = create_landmarker()
yolo_model = YOLO(YOLO_SEG_PATH)


# ----------------------------------
# VIDEO
# ----------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error al abrir el video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0

window_name = "DL Front - YOLO Seg + MediaPipe Skeleton"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)

print("Q para salir")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # ----------------------------------
    # YOLO SEGMENTACION
    # ----------------------------------

    results = yolo_model(frame, verbose=False)

    # Extraer centros de barras
    bar_centers = []
    result_obj = results[0]
    if result_obj.boxes is not None:
        for i in range(len(result_obj.boxes)):
            cls_id = int(result_obj.boxes.cls[i])
            if cls_id == 0:  # bar
                x1, y1, x2, y2 = map(int, result_obj.boxes.xyxy[i])
                cx = (x1 + x2) // 2
                cy = y2
                bar_centers.append((cx, cy))

    annotated_frame = result_obj.plot()

    # Linea horizontal entre 2 barras
    if len(bar_centers) == 2:
        bar_centers.sort(key=lambda p: p[0])
        lx, ly = bar_centers[0]
        rx, ry = bar_centers[1]
        line_y = (ly + ry) // 2
        cv2.line(annotated_frame, (lx, line_y), (rx, line_y), (0, 255, 255), 3)
        cv2.circle(annotated_frame, (lx, line_y), 5, (0, 255, 255), -1)
        cv2.circle(annotated_frame, (rx, line_y), 5, (0, 255, 255), -1)

    # ----------------------------------
    # MEDIAPIPE SKELETON
    # ----------------------------------

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

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        for idx in LANDMARKS:

            lm = landmarks[idx]

            if lm.visibility < CONF_THRESHOLD:
                continue

            x = int(lm.x * w)
            y = int(lm.y * h)

            cv2.circle(annotated_frame, (x,y), 5, (0,255,0), -1)

        for a,b in CONNECTIONS:

            la = landmarks[a]
            lb = landmarks[b]

            if la.visibility < CONF_THRESHOLD or lb.visibility < CONF_THRESHOLD:
                continue

            x1 = int(la.x * w)
            y1 = int(la.y * h)

            x2 = int(lb.x * w)
            y2 = int(lb.y * h)

            cv2.line(annotated_frame, (x1,y1), (x2,y2), (255,0,0), 3)

    cv2.imshow(window_name, annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_idx += 1


cap.release()
landmarker.close()
cv2.destroyAllWindows()
