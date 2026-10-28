"""Dibujo de las máscaras de barra, banco y persona sobre el frame."""

import cv2
import numpy as np

# BGR. La barra va en amarillo y más opaca: es lo que el juez necesita ver con claridad.
COLORS = {"bar": (0, 255, 255), "bench": (255, 0, 255), "person": (0, 165, 255)}
ALPHAS = {"bar": 0.65, "bench": 0.35, "person": 0.30}
DRAW_ORDER = ("bench", "person", "bar")     # la barra queda encima


def draw_segmentation(image, result, labels=True, alphas=None):
    """Copia de `image` con las máscaras de `result` rellenas y su contorno.

    Pensado para dibujarse sobre el frame ya anotado con el esqueleto: se pinta solo dentro
    de la caja de cada máscara y no toca nada más.
    """
    alphas = alphas or ALPHAS
    annotated = image.copy()
    height, width = annotated.shape[:2]
    for class_name in DRAW_ORDER:
        color = np.array(COLORS[class_name], dtype=np.float32)
        for instance in result.by_class(class_name):
            x1, y1 = max(int(np.floor(instance.bbox_xyxy[0])), 0), max(int(np.floor(instance.bbox_xyxy[1])), 0)
            x2, y2 = min(int(np.ceil(instance.bbox_xyxy[2])), width), min(int(np.ceil(instance.bbox_xyxy[3])), height)
            if x2 <= x1 or y2 <= y1:
                continue
            crop_mask = instance.mask[y1:y2, x1:x2]
            region = annotated[y1:y2, x1:x2]
            blended = region[crop_mask].astype(np.float32) * (1 - alphas[class_name]) + color * alphas[class_name]
            region[crop_mask] = blended.astype(np.uint8)
            if instance.polygon is not None:
                cv2.polylines(annotated, [instance.polygon.astype(np.int32)], True,
                              COLORS[class_name], 3 if class_name == "bar" else 2)
            if labels:
                cv2.putText(annotated, f"{class_name} {instance.confidence:.2f}", (x1, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[class_name], 2, cv2.LINE_AA)
    return annotated
