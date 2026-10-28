"""Tipos del módulo de segmentación (press de banca).

Los modelos YOLO de segmentación entregan tres clases: 0 bar, 1 bench, 2 person. El modelo
ya distingue barra, banco y cuerpo; aquí no se filtra ni reinterpreta ninguna.
"""

from dataclasses import dataclass

import numpy as np

CLASS_NAMES = {0: "bar", 1: "bench", 2: "person"}


@dataclass(frozen=True)
class Instance:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: np.ndarray       # (4,) float32 en píxeles
    mask: np.ndarray            # (H, W) bool, mismo tamaño que el frame
    polygon: np.ndarray = None  # (N, 2) float32 en píxeles, si el modelo lo entrega

    @property
    def area(self):
        return int(self.mask.sum())


@dataclass(frozen=True)
class SegmentationResult:
    instances: tuple
    image_shape: tuple          # (alto, ancho)

    def by_class(self, class_name):
        return [i for i in self.instances if i.class_name == class_name]

    def best(self, class_name):
        """Instancia de mayor confianza de esa clase, o None si no hay."""
        candidates = self.by_class(class_name)
        return max(candidates, key=lambda i: i.confidence) if candidates else None

    def has(self, class_name):
        return bool(self.by_class(class_name))
