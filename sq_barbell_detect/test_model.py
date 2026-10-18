"""Test rápido del modelo entrenado en un video SQ left."""
import cv2
from ultralytics import YOLO
from pathlib import Path

MODEL = Path(r"C:\Users\basti\MediapipePythonProjects\models\sq\sq_barbell_detect.pt")
VIDEO = Path(r"C:\Users\basti\MediapipePythonProjects\dataset\sq\left\sq_007.mp4")

if not MODEL.exists():
    print(f"Modelo no encontrado: {MODEL}")
    exit(1)

model = YOLO(str(MODEL))
cap = cv2.VideoCapture(str(VIDEO))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, imgsz=960, verbose=False)[0]
    annotated = results.plot()
    cv2.imshow("SQ Barbell Detect", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()