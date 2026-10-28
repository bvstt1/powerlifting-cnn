"""Lectura continua de una cámara en un hilo propio.

El hilo guarda siempre el frame más reciente. Una cámara lenta o con retraso nunca
bloquea a las demás y el consumidor nunca procesa imágenes antiguas acumuladas.
"""

import sys
import threading
import time
from dataclasses import dataclass

import numpy as np


class CameraError(RuntimeError):
    """No se pudo abrir una cámara o dejó de entregar imágenes."""


@dataclass(frozen=True)
class Frame:
    view: str
    image: np.ndarray       # BGR, tal como lo entrega OpenCV
    timestamp: float        # time.perf_counter() justo después de capturar el frame
    seq: int                # contador por cámara; crece con cada frame nuevo


def backend_flag(name):
    """Código de backend de OpenCV para el nombre usado en config.yaml."""
    import cv2

    flags = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}
    if name == "auto":
        return cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    return flags[name]


def open_capture(index, capture_config):
    """Abre cv2.VideoCapture con la resolución y los FPS del config."""
    import cv2

    capture = cv2.VideoCapture(index, backend_flag(capture_config.backend))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, capture_config.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_config.height)
    capture.set(cv2.CAP_PROP_FPS, capture_config.fps)
    return capture


def probe_cameras(capture_config, max_index=6):
    """Índices que abren y entregan un frame, con la resolución real obtenida."""
    import cv2

    found = []
    for index in range(max_index):
        capture = open_capture(index, capture_config)
        try:
            if capture.isOpened() and capture.grab():
                found.append({"index": index,
                              "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                              "fps": capture.get(cv2.CAP_PROP_FPS)})
        finally:
            capture.release()
    return found


class CameraReader:
    """Hilo lector de una cámara.

    `condition` es compartida por todos los lectores: se notifica con cada frame nuevo
    para que el consumidor espere en un único punto en lugar de sondear.
    """

    def __init__(self, view, index, capture, condition, max_failures=30):
        self.view = view
        self.index = index
        self._capture = capture
        self._condition = condition
        self._max_failures = max_failures
        self._frame = None
        self._seq = 0
        self._error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"camera-{view}", daemon=True)

    def start(self):
        if not self._capture.isOpened():
            self._capture.release()
            raise CameraError(f"No se pudo abrir la cámara {self.index} (vista {self.view})")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._capture.release()

    def latest(self):
        with self._condition:
            return self._frame

    @property
    def error(self):
        with self._condition:
            return self._error

    def _fail(self, message):
        with self._condition:
            self._error = CameraError(f"Cámara {self.index} (vista {self.view}): {message}")
            self._condition.notify_all()

    def _run(self):
        failures = 0
        while not self._stop.is_set():
            grabbed = self._capture.grab()
            # perf_counter y no monotonic: en Windows con Python < 3.13 monotonic tiene
            # resolución de ~15 ms, comparable a la tolerancia de sincronía.
            captured_at = time.perf_counter()
            ok, image = self._capture.retrieve() if grabbed else (False, None)
            if not ok or image is None:
                failures += 1
                if failures >= self._max_failures:
                    self._fail(f"dejó de entregar imágenes ({failures} lecturas fallidas seguidas)")
                    return
                time.sleep(0.005)
                continue
            failures = 0
            with self._condition:
                self._seq += 1
                self._frame = Frame(self.view, image, captured_at, self._seq)
                self._condition.notify_all()
