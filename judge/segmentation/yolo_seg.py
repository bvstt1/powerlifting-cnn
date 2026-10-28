"""Extractor con los modelos YOLO de segmentación de barra, banco y persona.

Inferencia con los parámetros de `extract_segmentation/bp/extract_segmentation_*_bp.py`:
imgsz=960, conf=0.25, iou=0.70, retina_masks=True (máscaras al tamaño del frame), sin
augment ni tracking. Cada instancia conserva clase, confianza, caja, máscara y polígono.
"""

import numpy as np

from judge.segmentation.base import CLASS_NAMES, Instance, SegmentationResult


def result_to_instances(result, image_shape):
    """Convierte un resultado de Ultralytics en SegmentationResult, con las mismas comprobaciones
    de correspondencia 1:1 y tamaño de máscara que el extractor de entrenamiento."""
    height, width = image_shape
    boxes = result.boxes
    count = len(boxes) if boxes is not None else 0
    if count == 0:
        return SegmentationResult((), (height, width))
    if result.masks is None or len(result.masks.data) != count:
        raise RuntimeError("YOLO: cajas y máscaras sin correspondencia 1:1")

    masks = result.masks.data.cpu().numpy() > 0.5
    if masks.shape[1:] != (height, width):
        raise RuntimeError(f"Máscara {masks.shape[1:]} distinta del frame {(height, width)}: verificar retina_masks")
    xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy().astype(np.float32)
    polygons = getattr(result.masks, "xy", None)

    instances = []
    for i in range(count):
        class_id = int(class_ids[i])
        if class_id not in CLASS_NAMES or str(result.names[class_id]) != CLASS_NAMES[class_id]:
            raise RuntimeError(f"Clase inesperada {class_id}")
        if not masks[i].any():
            continue                    # máscara vacía: nada que dibujar ni medir
        polygon = None
        if polygons is not None and i < len(polygons) and len(polygons[i]) >= 3:
            polygon = np.asarray(polygons[i], dtype=np.float32)
        instances.append(Instance(class_id, CLASS_NAMES[class_id], float(confidences[i]),
                                  xyxy[i], masks[i], polygon))
    return SegmentationResult(tuple(instances), (height, width))


class YoloSegmentationExtractor:
    """Un modelo de segmentación. Puede procesar varias imágenes en una sola llamada."""

    def __init__(self, model_path, imgsz=960, conf=0.25, iou=0.70, device=None, model=None):
        if model is None:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
        names = {int(k): str(v) for k, v in model.names.items()}
        if names != CLASS_NAMES:
            raise ValueError(f"{model_path}: clases {names}, se esperaban {CLASS_NAMES}")
        if device is None:
            import torch
            device = 0 if torch.cuda.is_available() else "cpu"
        self.model = model
        self.imgsz, self.conf, self.iou, self.device = imgsz, conf, iou, device

    def extract_many(self, images):
        """Lista de SegmentationResult, uno por imagen BGR."""
        if not images:
            return []
        results = self.model(list(images), imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                             retina_masks=True, augment=False, device=self.device, verbose=False)
        if len(results) != len(images):
            raise RuntimeError(f"YOLO devolvió {len(results)} resultados para {len(images)} imágenes")
        return [result_to_instances(r, image.shape[:2]) for r, image in zip(results, images)]

    def extract(self, image):
        return self.extract_many([image])[0]

    def close(self):
        pass
