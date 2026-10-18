"""Seguimiento exploratorio de barra SQ en píxeles del vídeo original.

Inspiración conceptual: Marticles/barbell-path-tracker (MIT, 2018): selección
manual de caja, seguimiento y trayectoria de centros. Implementación nueva:
CSRT OpenCV moderno, estados explícitos, huecos y controles temporales. Véase
display/sq/BAR_TRACKING_SQ.md y third_party/barbell-path-tracker-LICENSE.
"""
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def create_csrt():
    factory = getattr(cv2, "TrackerCSRT_create", None)
    if factory is None and hasattr(cv2, "legacy"):
        factory = getattr(cv2.legacy, "TrackerCSRT_create", None)
    if factory is None:
        raise RuntimeError(
            "OpenCV cargado no ofrece TrackerCSRT_create (ni cv2.legacy). "
            "Las ruedas opencv-python y opencv-contrib-python comparten cv2; "
            "se necesita que la versión contrib activa incluya CSRT."
        )
    return factory()


class VideoViewport:
    """Vídeo ajustado sin deformación a un rectángulo con bandas negras."""

    def __init__(self, video_width, video_height, left, top, width, height):
        if min(video_width, video_height, width, height) <= 0:
            raise ValueError("Dimensiones de vídeo/display inválidas")
        self.video_width = video_width
        self.video_height = video_height
        self.target = (left, top, width, height)
        factor = min(width / video_width, height / video_height)
        self.width = max(1, min(width, int(round(video_width * factor))))
        self.height = max(1, min(height, int(round(video_height * factor))))
        self.left = left + (width - self.width) // 2
        self.top = top + (height - self.height) // 2
        self.sx = self.width / video_width
        self.sy = self.height / video_height

    def to_video(self, x, y):
        """None si el click fue sobre banda negra o panel lateral."""
        if not (self.left <= x < self.left + self.width and
                self.top <= y < self.top + self.height):
            return None
        return ((x - self.left) / self.sx, (y - self.top) / self.sy)

    def to_display(self, x, y):
        return (int(round(self.left + x * self.sx)),
                int(round(self.top + y * self.sy)))

    def roi_from_display(self, start, end):
        a = self.to_video(*start)
        b = self.to_video(*end)
        if a is None or b is None:
            return None
        x0 = max(0, int(math.floor(min(a[0], b[0]))))
        y0 = max(0, int(math.floor(min(a[1], b[1]))))
        x1 = min(self.video_width, int(math.ceil(max(a[0], b[0]))))
        y1 = min(self.video_height, int(math.ceil(max(a[1], b[1]))))
        return (x0, y0, x1 - x0, y1 - y0) if x1-x0 >= 4 and y1-y0 >= 4 else None


class BarTracker:
    """Una sesión por vídeo: no mezcla frames, saltos o fallos con observaciones."""

    def __init__(self, fps, tracker_factory=create_csrt):
        self.fps = fps if fps > 0 else 30.0
        self.tracker_factory = tracker_factory
        self.enabled = False
        self.reset()

    def reset(self):
        self.tracker = None
        self.state = "sin inicializar"
        self.last_index = None
        self.box = None
        self.records = []

    def set_enabled(self, enabled):
        if self.enabled != enabled:
            self.reset()
        self.enabled = enabled

    def change_video(self, fps):
        self.reset()
        self.fps = fps if fps > 0 else 30.0

    def _row(self, index, box, status):
        x, y, w, h = box if box is not None else (None, None, None, None)
        return {"frame": int(index), "time_s": index / self.fps,
                "center_x": x + w / 2 if box is not None else None,
                "center_y": y + h / 2 if box is not None else None,
                "box_x": x, "box_y": y, "box_w": w, "box_h": h,
                "status": status}

    @staticmethod
    def valid_box(box, frame):
        x, y, w, h = box
        height, width = frame.shape[:2]
        return (np.isfinite(box).all() and w >= 4 and h >= 4 and
                x >= 0 and y >= 0 and x+w <= width and y+h <= height)

    def initialize(self, frame, index, box):
        if not self.enabled:
            raise RuntimeError("Activar tracking antes de seleccionar ROI")
        if not self.valid_box(box, frame):
            raise ValueError("ROI fuera del vídeo o demasiado pequeña")
        self.reset()
        tracker = self.tracker_factory()
        result = tracker.init(frame, tuple(map(int, box)))
        if result is False:
            self.state = "seguimiento perdido"
            self.last_index = index
            self.records.append(self._row(index, None, "perdido"))
            return False
        self.tracker = tracker
        self.state = "siguiendo"
        self.last_index = index
        self.box = tuple(map(float, box))
        self.records.append(self._row(index, self.box, "siguiendo"))
        return True

    def update(self, frame, index):
        if not self.enabled or self.state != "siguiendo":
            return False
        if index != self.last_index + 1:
            self.reset()  # retroceso/salto: estado de CSRT inválido
            return False
        ok, box = self.tracker.update(frame)
        self.last_index = index
        if not ok or not self.valid_box(box, frame):
            self.state = "seguimiento perdido"
            self.tracker = None
            self.box = None
            self.records.append(self._row(index, None, "perdido"))
            return False
        self.box = tuple(map(float, box))
        self.records.append(self._row(index, self.box, "siguiendo"))
        return True

    def csv_export(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=("frame", "time_s", "center_x",
                "center_y", "box_x", "box_y", "box_w", "box_h", "status"))
            writer.writeheader()
            writer.writerows(self.records)
        return path

    def current_center(self, index):
        """Centro SOLO de una caja observada en este frame (nunca extrapolado)."""
        if self.state != "siguiendo" or self.last_index != index or self.box is None:
            return None
        x, y, w, h = self.box
        return (x + w / 2, y + h / 2)

    def draw(self, canvas, viewport, index):
        magenta = (210, 60, 255)  # distinto de esqueleto verde y landmarks cian
        previous = None
        previous_index = None
        for row in self.records:
            if row["frame"] > index:
                break
            if row["status"] != "siguiendo":
                previous = None
                continue
            point = viewport.to_display(row["center_x"], row["center_y"])
            if previous is not None and row["frame"] == previous_index + 1:
                cv2.line(canvas, previous, point, magenta, 2)
            previous, previous_index = point, row["frame"]
        center = self.current_center(index)
        if center is not None:
            x, y, w, h = self.box
            a = viewport.to_display(x, y)
            b = viewport.to_display(x+w, y+h)
            cv2.rectangle(canvas, a, b, magenta, 2)
            cx, cy = viewport.to_display(*center)
            # Diana contrastada: la marca central blanca es el centro medido
            # de la caja en ESTE frame, no un landmark del esqueleto.
            cv2.circle(canvas, (cx,cy), 20, (0,0,0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx,cy), 16, (0,255,255), 3, cv2.LINE_AA)
            cv2.circle(canvas, (cx,cy), 10, magenta, -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx,cy), 3, (255,255,255), -1, cv2.LINE_AA)
            for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
                start = (cx + 20*dx, cy + 20*dy)
                end = (cx + 28*dx, cy + 28*dy)
                cv2.line(canvas, start, end, (255,255,255), 2, cv2.LINE_AA)

            label = f"BARRA ({center[0]:.0f}, {center[1]:.0f})"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .6, 2)
            lx = cx + 28 if cx + text_w + 36 < viewport.left + viewport.width else cx-text_w-36
            lx = max(viewport.left + 2, min(lx, viewport.left+viewport.width-text_w-12))
            ly = cy-22 if cy-text_h-28 > viewport.top else cy+text_h+35
            ly = max(viewport.top+text_h+6, min(ly,viewport.top+viewport.height-8))
            cv2.rectangle(canvas, (lx-5,ly-text_h-5), (lx+text_w+5,ly+baseline+5),
                          (0,0,0), -1)
            cv2.putText(canvas, label, (lx,ly), cv2.FONT_HERSHEY_SIMPLEX,
                        .6, (0,255,255), 2, cv2.LINE_AA)


def draw_vertical_plot(canvas, tracker, current_index, total_frames, frame_height,
                       rect, smooth=False):
    """Curva ORIGINAL (magenta); EMA separada opcional (naranja), sin sustituirla."""
    x0, y0, width, height = rect
    cv2.rectangle(canvas, (x0,y0), (x0+width,y0+height), (100,100,100), 1)
    cv2.putText(canvas, "Barra y (px): abajo +", (x0+6,y0+21),
                cv2.FONT_HERSHEY_SIMPLEX, .47, (230,230,230), 1)
    if total_frames <= 1:
        return
    marker = x0 + int(round(current_index * width / (total_frames-1)))
    cv2.line(canvas, (marker,y0+26), (marker,y0+height), (110,110,110), 1)
    previous = None
    smooth_prev = None
    ema = None
    previous_index = None
    for row in tracker.records:
        if row["frame"] > current_index:
            break
        if row["status"] != "siguiendo":
            previous = smooth_prev = ema = None
            continue
        px = x0 + int(round(row["frame"] * width / (total_frames-1)))
        py = y0 + 30 + int(round(row["center_y"] / frame_height * (height-32)))
        py = max(y0+30, min(y0+height-1,py))
        if previous is not None and row["frame"] == previous_index + 1:
            cv2.line(canvas, previous, (px,py), (210,60,255), 2)
        if smooth:
            ema = row["center_y"] if ema is None else .30*row["center_y"] + .70*ema
            sy = y0+30+int(round(ema/frame_height*(height-32)))
            sy = max(y0+30, min(y0+height-1,sy))
            if smooth_prev is not None and row["frame"] == previous_index + 1:
                cv2.line(canvas,smooth_prev,(px,sy),(0,165,255),1)
            smooth_prev = (px,sy)
        previous, previous_index = (px,py), row["frame"]
