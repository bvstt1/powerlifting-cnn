"""Dibujo del esqueleto sobre el frame de una vista."""

import cv2

LINE_COLOR = (255, 0, 0)        # BGR, como los visores del repo
POINT_COLOR = (0, 255, 0)
LABEL_COLOR = (0, 255, 255)


def draw_skeleton(image, result, skeleton, min_confidence=0.5, labels=False):
    """Copia de `image` con los puntos y conexiones del modelo `skeleton` (SkeletonModel).

    Solo se dibujan los puntos visibles con confianza >= `min_confidence`; una conexión
    exige que ambos extremos lo sean. El punto 24 de MediaPipe frontal (distancia
    barra-rodilla) es una medida, no una posición, y no se dibuja.
    """
    annotated = image.copy()
    height, width = annotated.shape[:2]
    pixels = result.to_pixels(width, height)
    visible = result.visible(min_confidence)
    drawable = skeleton.drawable_points or skeleton.num_points
    visible[drawable:] = False

    for a, b in skeleton.connections:
        if visible[a] and visible[b]:
            cv2.line(annotated, tuple(int(v) for v in pixels[a]), tuple(int(v) for v in pixels[b]),
                     LINE_COLOR, 3)
    for index in range(drawable):
        if not visible[index]:
            continue
        center = (int(pixels[index][0]), int(pixels[index][1]))
        cv2.circle(annotated, center, 6, POINT_COLOR, -1)
        if labels:
            name = skeleton.points[index] if index < len(skeleton.points) else str(index)
            cv2.putText(annotated, f"{name} {result.points[index][2]:.2f}", (center[0] + 8, center[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, LABEL_COLOR, 2, cv2.LINE_AA)
    return annotated
