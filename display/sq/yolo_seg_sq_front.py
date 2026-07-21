from ultralytics import YOLO
import cv2

# Cargar modelo
model = YOLO("../../models/sq_front_seg_v1.pt")

# Ruta del video
video_path = r"C:\Users\basti\MediapipePythonProjects\dataset\sq\front\sq_203.mp4"

# Abrir video
cap = cv2.VideoCapture(video_path)

# Verificar apertura
if not cap.isOpened():
    print("Error al abrir el video")
    exit()

# Crear ventana redimensionable
window_name = "YOLO Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Tamaño de la ventana
cv2.resizeWindow(window_name, 800, 600)

while True:
    ret, frame = cap.read()

    # Fin del video
    if not ret:
        break

    # Inferencia YOLO
    results = model(frame)

    # Dibujar detecciones
    annotated_frame = results[0].plot()

    # Mostrar frame
    cv2.imshow(window_name, annotated_frame)

    # Presiona Q para salir
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()