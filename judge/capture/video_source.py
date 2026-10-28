"""Fuente de frames desde 3 videos grabados, con la misma interfaz que MultiCameraCapture.

Sirve para evaluar un levantamiento ya grabado sin cámaras. Los videos avanzan en paralelo,
un frame por vista; la lectura termina cuando el más corto se acaba. Las marcas de tiempo se
derivan del índice de frame y de los FPS de cada video, así que las vistas quedan sincronizadas
por posición (frame N de cada video), no por reloj.
"""

from pathlib import Path

import cv2

from judge.capture.camera import CameraError, Frame
from judge.capture.multi_camera import FrameSet
from judge.config import VIEWS


class VideoSource:
    def __init__(self, paths, stride=1):
        """paths: {vista: ruta}. stride=N entrega 1 de cada N frames."""
        if set(paths) != set(VIEWS):
            raise CameraError(f"Se requieren videos para {VIEWS}, recibidos {sorted(paths)}")
        if stride < 1:
            raise ValueError("stride debe ser >= 1")
        self.paths = {view: Path(path) for view, path in paths.items()}
        self.stride = stride
        self._captures = {}
        self._fps = {}
        self._index = 0
        self.delivered_sets = 0
        self.dropped_frames = 0

    def start(self):
        for view, path in self.paths.items():
            if not path.is_file():
                self.stop()
                raise CameraError(f"No existe el video de la vista {view}: {path}")
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                capture.release()
                self.stop()
                raise CameraError(f"No se pudo abrir el video de la vista {view}: {path}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            self._captures[view] = capture
            self._fps[view] = fps if fps and fps > 0 else 30.0
        return self

    def stop(self):
        for capture in self._captures.values():
            capture.release()
        self._captures = {}

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def read(self, timeout=None):
        """Siguiente conjunto de frames, o None cuando algún video termina."""
        if not self._captures:
            raise CameraError("La fuente de video no está iniciada")
        frames = {}
        for view, capture in self._captures.items():
            for skipped in range(self.stride - 1):
                if not capture.grab():
                    return None
            ok, image = capture.read()
            if not ok:
                return None
            frames[view] = Frame(view, image, self._index / self._fps[view], self._index)
        self._index += self.stride
        self.delivered_sets += 1
        return FrameSet(frames)
