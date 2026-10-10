from ultralytics import YOLO
import cv2

# -------------------------
# CONFIG
# -------------------------

model = YOLO("../../../models/bp/bp_side_skeleton_v2.pt")

video_path = r"/dataset/bp/right/bp_549.mp4"

# Conexiones skeleton (ajusta según tu modelo)
SKELETON = [
    (0, 1),
    (0, 2),
    (2, 3),
    (3, 5),
    (5, 4),
    (1, 6),
    (6, 7),
    (7, 1)
]

# -------------------------
# VIDEO
# -------------------------

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error al abrir video")
    exit()

window_name = "Pose Skeleton - Ver Indices"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 900, 600)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)
    annotated = frame.copy()

    if len(results) > 0 and results[0].keypoints is not None:
        # Dependiendo de la versión de Ultralytics, keypoints.xy puede venir en shape (N, K, 2)
        keypoints = results[0].keypoints.xy.cpu().numpy()

        for person in keypoints:

            # -------------------------
            # DIBUJAR LÍNEAS PRIMERO (opcional para que no tapen los textos)
            # -------------------------
            for p1, p2 in SKELETON:
                # Asegurarse de que los índices existan en la persona detectada
                if p1 < len(person) and p2 < len(person):
                    x1, y1 = person[p1]
                    x2, y2 = person[p2]

                    # Ignorar puntos con coordenadas (0,0) si no fueron detectados
                    if x1 != 0 and y1 != 0 and x2 != 0 and y2 != 0:
                        cv2.line(
                            annotated,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (255, 0, 0),
                            3
                        )

            # -------------------------
            # DIBUJAR PUNTOS Y NÚMEROS
            # -------------------------
            for idx, (x, y) in enumerate(person):
                if x == 0 and y == 0:
                    continue  # Salta puntos no detectados

                ix, iy = int(x), int(y)

                # Dibujar círculo del punto
                cv2.circle(
                    annotated,
                    (ix, iy),
                    6,
                    (0, 255, 0),
                    -1
                )

                # Dibujar el número del índice al lado del punto
                cv2.putText(
                    annotated,
                    str(idx),
                    (ix + 8, iy - 8),  # Desplazado un poco para que sea legible
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,  # Escala de la fuente
                    (0, 0, 255),  # Color rojo para el texto
                    2,  # Grosor de la línea
                    cv2.LINE_AA
                )

    cv2.imshow(window_name, annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()