"""Tipos comunes de los extractores de keypoints.

El formato de salida es el mismo con el que se extrajeron los datos de entrenamiento
(`extract_keypoints/*`): un arreglo (K, 3) float32 con x, y normalizados al tamaño de la
imagen y un tercer canal (confianza YOLO o visibility MediaPipe). Sin persona detectada,
todos los valores son NaN.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KeypointResult:
    points: np.ndarray      # (K, 3) float32

    @property
    def detected(self):
        """True si hay al menos un punto con posición."""
        return bool(np.isfinite(self.points[:, :2]).all(axis=1).any())

    def visible(self, min_confidence=0.0):
        """Máscara (K,) de puntos utilizables.

        YOLO deja (0, 0) con confianza 0 en los puntos que no detecta; esos puntos
        tampoco son visibles aunque sus valores sean finitos.
        """
        xy, conf = self.points[:, :2], self.points[:, 2]
        finite = np.isfinite(xy).all(axis=1) & np.isfinite(conf)
        missing = (xy == 0).all(axis=1) & (conf == 0)
        return finite & ~missing & (conf >= min_confidence)

    def to_pixels(self, width, height):
        """Posiciones en píxeles (K, 2); NaN donde no hay punto."""
        return self.points[:, :2] * np.array([width, height], dtype=np.float32)


def empty_points(num_points):
    return np.full((num_points, 3), np.nan, dtype=np.float32)
