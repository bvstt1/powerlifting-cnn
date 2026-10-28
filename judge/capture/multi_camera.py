"""Captura simultánea de las 3 vistas y emparejado de frames por instante.

Las cámaras no comparten reloj de hardware. El emparejado se hace por software con la
marca de tiempo de captura: un conjunto de frames solo se entrega si los frames de las
3 vistas fueron capturados con una diferencia de `max_skew_ms` como máximo. A 30 FPS
el periodo de un frame es 33 ms, por eso el valor por defecto del config es 40 ms.
"""

import threading
import time
from dataclasses import dataclass

from judge.capture.camera import CameraError, CameraReader, open_capture
from judge.config import VIEWS


@dataclass(frozen=True)
class FrameSet:
    """Un frame por vista, capturados dentro de la tolerancia de sincronía."""
    frames: dict

    @property
    def timestamp(self):
        return max(f.timestamp for f in self.frames.values())

    @property
    def skew_ms(self):
        times = [f.timestamp for f in self.frames.values()]
        return (max(times) - min(times)) * 1000.0

    def images(self):
        return {view: frame.image for view, frame in self.frames.items()}


def select_set(latest, delivered_seq, max_skew_s):
    """Decide qué hacer con los frames más recientes de cada vista.

    Devuelve ("wait", None) si falta algún frame nuevo, ("deliver", FrameSet) si hay
    un conjunto sincronizado, o ("skewed", view) con la vista cuyo frame es el más
    antiguo y debe descartarse para esperar uno más reciente.
    """
    if any(latest.get(view) is None or latest[view].seq <= delivered_seq[view] for view in VIEWS):
        return "wait", None
    frames = {view: latest[view] for view in VIEWS}
    times = {view: frame.timestamp for view, frame in frames.items()}
    if max(times.values()) - min(times.values()) <= max_skew_s:
        return "deliver", FrameSet(frames)
    return "skewed", min(times, key=times.get)


class MultiCameraCapture:
    """Abre las 3 cámaras del config y entrega conjuntos de frames sincronizados.

    Uso:
        with MultiCameraCapture(config.capture) as capture:
            frame_set = capture.read(timeout=1.0)   # FrameSet o None si expira
    """

    def __init__(self, capture_config, capture_factory=None, max_failures=30):
        self.config = capture_config
        self._factory = capture_factory or open_capture
        self._max_failures = max_failures
        self._condition = threading.Condition()
        self._readers = {}
        self._delivered_seq = {view: 0 for view in VIEWS}
        self.delivered_sets = 0
        self.dropped_frames = 0

    def start(self):
        if self._readers:
            raise CameraError("La captura ya está iniciada")
        try:
            for view in VIEWS:
                index = self.config.cameras[view]
                reader = CameraReader(view, index, self._factory(index, self.config),
                                      self._condition, self._max_failures)
                reader.start()
                self._readers[view] = reader
        except Exception:
            self.stop()
            raise
        return self

    def stop(self):
        for reader in self._readers.values():
            reader.stop()
        self._readers = {}

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def _check_errors(self):
        for reader in self._readers.values():
            if reader.error is not None:
                raise reader.error

    def read(self, timeout=1.0):
        """Siguiente conjunto sincronizado, o None si no llega en `timeout` segundos."""
        if not self._readers:
            raise CameraError("La captura no está iniciada")
        max_skew_s = self.config.max_skew_ms / 1000.0
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                self._check_errors()
                latest = {view: reader._frame for view, reader in self._readers.items()}
                action, payload = select_set(latest, self._delivered_seq, max_skew_s)
                if action == "deliver":
                    for view, frame in payload.frames.items():
                        self._delivered_seq[view] = frame.seq
                    self.delivered_sets += 1
                    return payload
                if action == "skewed":
                    # El frame más viejo no tiene pareja: se descarta y se espera uno nuevo.
                    self._delivered_seq[payload] = latest[payload].seq
                    self.dropped_frames += 1
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
