"""Grabadora de un levantamiento, controlada a mano por el juez.

El juez decide cuándo empieza y cuándo termina el levantamiento (una tecla, que conecta el
pipeline principal). Mientras graba, cada conjunto de keypoints se agrega a la secuencia de su
vista; las vistas son independientes y pueden acumular un número distinto de frames.
"""

import time

import numpy as np

from judge.config import VIEWS
from judge.evaluation.base import Recording

IDLE = "idle"
RECORDING = "recording"


class RecorderError(RuntimeError):
    """Operación inválida para el estado actual de la grabadora."""


class LiftRecorder:
    def __init__(self, num_points, max_frames=6000, clock=time.perf_counter):
        """num_points: {vista: K} con el número de puntos de cada vista."""
        if set(num_points) != set(VIEWS):
            raise ValueError(f"num_points debe cubrir exactamente {VIEWS}")
        self._num_points = dict(num_points)
        self._max_frames = max_frames
        self._clock = clock
        self._reset()
        self.state = IDLE

    def _reset(self):
        self._frames = {view: [] for view in VIEWS}
        self._started_at = None
        self._truncated = False

    @property
    def recording(self):
        return self.state == RECORDING

    def frame_counts(self):
        return {view: len(frames) for view, frames in self._frames.items()}

    def start(self):
        if self.recording:
            raise RecorderError("Ya hay un levantamiento en grabación")
        self._reset()
        self._started_at = self._clock()
        self.state = RECORDING

    def add(self, results):
        """Agrega los keypoints de un instante: {vista: KeypointResult}. Devuelve False si no graba."""
        if not self.recording:
            return False
        for view, result in results.items():
            if view not in self._frames:
                raise RecorderError(f"Vista desconocida: {view}")
            points = np.asarray(result.points, dtype=np.float32)
            if points.shape != (self._num_points[view], 3):
                raise RecorderError(f"{view}: se esperaban {self._num_points[view]} puntos, recibidos {points.shape}")
            if len(self._frames[view]) >= self._max_frames:
                self._truncated = True
                continue
            self._frames[view].append(points)
        return True

    def stop(self):
        """Termina el levantamiento y devuelve la grabación."""
        if not self.recording:
            raise RecorderError("No hay un levantamiento en grabación")
        duration = self._clock() - self._started_at
        sequences = {view: (np.stack(frames) if frames
                            else np.empty((0, self._num_points[view], 3), dtype=np.float32))
                     for view, frames in self._frames.items()}
        recording = Recording(sequences, duration, self._truncated)
        self._reset()
        self.state = IDLE
        return recording

    def cancel(self):
        """Descarta el levantamiento en curso (por ejemplo, un intento mal iniciado)."""
        if not self.recording:
            raise RecorderError("No hay un levantamiento en grabación")
        self._reset()
        self.state = IDLE
