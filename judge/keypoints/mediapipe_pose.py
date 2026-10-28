"""Extractor MediaPipe del peso muerto frontal (25 puntos).

Reproduce `extract_keypoints/dl/extract_keypoints_mediapipe_dl.py` (vista frontal), que es
el formato con el que se entrenó la red de DL frontal: los 22 landmarks corporales sin
rostro (11..32), el centroide de cada mano y la distancia horizontal barra-rodilla.
Cada fila es (x, y, visibility).
"""

import numpy as np

from judge.keypoints.base import KeypointResult, empty_points

FACE_LANDMARKS = frozenset(range(11))
LEFT_HAND = (15, 17, 19, 21)
RIGHT_HAND = (16, 18, 20, 22)
LEFT_KNEE, RIGHT_KNEE = 25, 26
HAND_Y_OFFSET = 19 / 720.0      # desplazamiento fijo usado al extraer los datos de entrenamiento
VISIBILITY_MIN = 0.5
NUM_POINTS = 25


def landmarks_to_points(landmarks):
    """25 puntos a partir de los 33 landmarks de MediaPipe (lista con .x, .y, .visibility)."""
    rows = [[lm.x, lm.y, lm.visibility] for idx, lm in enumerate(landmarks) if idx not in FACE_LANDMARKS]

    left = [(landmarks[i].x, landmarks[i].y) for i in LEFT_HAND if landmarks[i].visibility >= VISIBILITY_MIN]
    if left:
        left_x = float(sum(p[0] for p in left) / len(left))
        left_y = float(sum(p[1] for p in left) / len(left)) + HAND_Y_OFFSET
        rows.append([left_x, left_y, 1.0])
    else:
        rows.append([np.nan, np.nan, 0.0])

    right = [(landmarks[i].x, landmarks[i].y) for i in RIGHT_HAND if landmarks[i].visibility >= VISIBILITY_MIN]
    if right:
        right_x = float(sum(p[0] for p in right) / len(right))
        right_y = float(sum(p[1] for p in right) / len(right)) + HAND_Y_OFFSET
        rows.append([right_x, right_y, 1.0])
    else:
        rows.append([np.nan, np.nan, 0.0])

    if (left and right and landmarks[LEFT_KNEE].visibility >= VISIBILITY_MIN
            and landmarks[RIGHT_KNEE].visibility >= VISIBILITY_MIN):
        bar_x = (left_x + right_x) / 2.0
        rows.append([float(bar_x - landmarks[LEFT_KNEE].x), float(bar_x - landmarks[RIGHT_KNEE].x), 1.0])
    else:
        rows.append([np.nan, np.nan, 0.0])

    return np.array(rows, dtype=np.float32)


class MediaPipeFrontDeadliftExtractor:
    """Landmarker de MediaPipe en modo video. Es con estado: usar una instancia por cámara."""

    def __init__(self, model_path, landmarker=None):
        if landmarker is None:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
            )
            landmarker = vision.PoseLandmarker.create_from_options(options)
        self._landmarker = landmarker
        self._last_ms = -1
        self.num_points = NUM_POINTS

    def extract(self, image, timestamp=None):
        """`timestamp` en segundos (reloj creciente); MediaPipe exige milisegundos estrictamente crecientes."""
        import cv2
        import mediapipe as mp

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        wanted = int(round(timestamp * 1000)) if timestamp is not None else self._last_ms + 33
        self._last_ms = max(wanted, self._last_ms + 1)
        result = self._landmarker.detect_for_video(mp_image, self._last_ms)
        if not result.pose_landmarks:
            return KeypointResult(empty_points(NUM_POINTS))
        return KeypointResult(landmarks_to_points(result.pose_landmarks[0]))

    def extract_many(self, images, timestamps=None):
        timestamps = timestamps or [None] * len(images)
        return [self.extract(image, ts) for image, ts in zip(images, timestamps)]

    def close(self):
        self._landmarker.close()
