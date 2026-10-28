"""Extractor de keypoints con los modelos YOLO de skeleton propios.

Reproduce `extract_keypoints/{bp,sq,dl}/extract_keypoints_yolo_*.py`: se toma la persona
con mayor confianza de caja, coordenadas `xyn` y confianza por punto, con `imgsz=960`.
Los puntos que el modelo no detecta quedan como los entrega YOLO ((0, 0) con confianza 0).
"""

import numpy as np

from judge.keypoints.base import KeypointResult, empty_points


def result_to_points(result, num_points):
    """Convierte un resultado de Ultralytics en (K, 3), igual que el extractor de entrenamiento."""
    points = empty_points(num_points)
    if result.keypoints is None or len(result.keypoints) == 0:
        return points

    if result.boxes is not None and len(result.boxes) > 0:
        person = int(np.argmax(result.boxes.conf.cpu().numpy()))
    else:
        person = 0

    coordinates = result.keypoints.xyn[person].cpu().numpy().astype(np.float32)
    if coordinates.shape != (num_points, 2):
        raise RuntimeError(f"Se esperaban {num_points} puntos, recibidos: {coordinates.shape}")

    if result.keypoints.conf is None:
        confidence = np.ones(num_points, dtype=np.float32)
    else:
        confidence = result.keypoints.conf[person].cpu().numpy().astype(np.float32)
    if confidence.shape != (num_points,):
        raise RuntimeError(f"Se esperaban {num_points} confianzas, recibidas: {confidence.shape}")

    points[:, :2] = coordinates
    points[:, 2] = confidence
    return points


class YoloKeypointExtractor:
    """Un modelo YOLO de skeleton. Puede procesar varias imágenes en una sola llamada."""

    def __init__(self, model_path, num_points, imgsz=960, device=None, model=None):
        if model is None:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
        shape = model.model.yaml.get("kpt_shape")
        if not shape or len(shape) != 2 or shape[1] != 3 or shape[0] != num_points:
            raise ValueError(f"{model_path}: kpt_shape={shape}, se esperaba [{num_points}, 3]")
        if device is None:
            import torch
            device = 0 if torch.cuda.is_available() else "cpu"
        self.model = model
        self.num_points = num_points
        self.imgsz = imgsz
        self.device = device

    def extract_many(self, images, timestamps=None):
        """Lista de KeypointResult, uno por imagen BGR."""
        if not images:
            return []
        results = self.model(list(images), imgsz=self.imgsz, device=self.device, verbose=False)
        if len(results) != len(images):
            raise RuntimeError(f"YOLO devolvió {len(results)} resultados para {len(images)} imágenes")
        return [KeypointResult(result_to_points(r, self.num_points)) for r in results]

    def extract(self, image, timestamp=None):
        return self.extract_many([image])[0]

    def close(self):
        pass
