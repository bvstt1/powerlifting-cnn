import gc
import torch
from pathlib import Path
from roboflow import Roboflow
from ultralytics import YOLO

# =========================
# DESCARGAR DATASET
# =========================

rf = Roboflow(api_key="BZbBb9akBQXA6mYdJDk5")
project = rf.workspace("franciscos-workspace-bnicm").project("dl_side_skeleton")
version = project.version(1)
dataset = version.download("yolov8")

# Usar directamente el data.yaml descargado por Roboflow
DATA_YAML = str(Path(dataset.location) / "data.yaml")


# =========================
# MODELOS
# =========================

models = [
    "yolov8n-pose.pt",
    "yolov8s-pose.pt",
    "yolov8m-pose.pt",

    "yolo11n-pose.pt",
    "yolo11s-pose.pt",
    "yolo11m-pose.pt",

    "yolo26n-pose.pt",
    "yolo26s-pose.pt",
    "yolo26m-pose.pt",
]


# =========================
# PARÁMETROS COMUNES
# =========================

COMMON_ARGS = dict(
    data=DATA_YAML,

    # Entrenamiento
    epochs=200,
    patience=30,

    # GPU / imágenes
    imgsz=960,
    batch=32,
    device=0,

    # Optimización
    optimizer="auto",
    workers=12,

    # Mejor usar disk con 32 GB de RAM
    cache="disk",

    amp=True,
    cos_lr=True,

    # Augmentations geométricas
    degrees=5,
    translate=0.05,
    scale=0.25,
    shear=0.0,
    perspective=0.0,

    # Augmentations desactivadas para Pose
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    fliplr=0.0,
    flipud=0.0,

    # Color / iluminación
    hsv_h=0.015,
    hsv_s=0.35,
    hsv_v=0.25,

    # Reproducibilidad
    seed=42,
    deterministic=True,

    # Resultados
    save=True,
    save_period=10,
    plots=True,

    project="runs/dl_side_comparison",
)


# =========================
# ENTRENAMIENTO AUTOMÁTICO
# =========================

for i, model_name in enumerate(models, start=1):

    experiment_name = model_name.replace(".pt", "")

    print("\n" + "=" * 70)
    print(f"ENTRENAMIENTO {i}/{len(models)}")
    print(f"Modelo: {model_name}")
    print("=" * 70 + "\n")

    try:
        # Si no existe localmente, Ultralytics lo descarga automáticamente
        model = YOLO(model_name)

        # Entrenar
        results = model.train(
            **COMMON_ARGS,
            name=experiment_name
        )

        print(f"\n✓ Entrenamiento completado: {model_name}")

    except Exception as e:

        print(f"\n✗ Error entrenando {model_name}")
        print(e)

    finally:
        # Liberar memoria antes del siguiente modelo
        try:
            del model
        except:
            pass

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\nMemoria liberada. Continuando con el siguiente modelo...\n")


print("\n" + "=" * 70)
print("TODOS LOS ENTRENAMIENTOS HAN FINALIZADO")
print("=" * 70)