from ultralytics import YOLO
import cv2

# -------------------------
# CONFIG
# -------------------------

model = YOLO("../../models/bp_front_skeleton_v6.pt")

test = r"C:\Users\basti\Desktop\test.mp4"
video_path = r"C:\Users\basti\MediapipePythonProjects\dataset\bp\front\bp_125.mp4"

# Conexiones skeleton
SKELETON = [
    (0, 1),  # codo izq -> hombro izq

    (1, 2),  # hombro izq -> hombro der

    (2,3), # hombro der -> codo der

    (3, 4),# codo der -> muñeca der
    (0, 5) # codo izq -> muñeca der

]

# -------------------------
# VIDEO
# -------------------------

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error al abrir video")
    exit()

window_name = "Pose Skeleton"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 900, 600)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    annotated = frame.copy()

    if len(results) > 0 and results[0].keypoints is not None:

        keypoints = results[0].keypoints.xy.cpu().numpy()

        for person in keypoints:

            # -------------------------
            # DIBUJAR PUNTOS
            # -------------------------

            for x, y in person:

                cv2.circle(
                    annotated,
                    (int(x), int(y)),
                    6,
                    (0, 255, 0),
                    -1
                )

            # -------------------------
            # DIBUJAR LINEAS
            # -------------------------

            for p1, p2 in SKELETON:

                x1, y1 = person[p1]
                x2, y2 = person[p2]

                cv2.line(
                    annotated,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255, 0, 0),
                    3
                )

    cv2.imshow(window_name, annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()