"""Entrena YOLOv26 detección de barra SQ lateral."""
from ultralytics import YOLO

DATA_YAML = r"C:\Users\basti\MediapipePythonProjects\sq_barbell_detect\data.yaml"
MODEL_OUT = r"C:\Users\basti\MediapipePythonProjects\models\sq\sq_barbell_detect.pt"

if __name__ == "__main__":
    model = YOLO("yolo26n.pt")  # nano: rápido, bueno para objeto único
    results = model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=960,
        batch=8,
        device=0,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        patience=20,
        project="runs/detect",
        name="sq_barbell",
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    best.replace(MODEL_OUT)
    print(f"Modelo guardado en: {MODEL_OUT}")