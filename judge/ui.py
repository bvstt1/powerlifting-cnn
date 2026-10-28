"""Composición de la ventana del juez: mosaico de las 3 vistas, estado y decisión.

Solo funciones puras sobre imágenes; la ventana y el teclado los maneja `judge.main`.
"""

import cv2
import numpy as np

from judge.config import VIEWS

VIEW_TITLES = {"front": "FRONTAL", "left": "LATERAL IZQ.", "right": "LATERAL DER."}
BANNER_COLORS = {"valid": (60, 160, 60), "invalid": (50, 50, 200), "manual_review": (0, 140, 230)}   # BGR
STATE_COLORS = {"idle": (200, 200, 200), "recording": (0, 0, 255)}


def _tile(image, title, height):
    if image is None:
        tile = np.zeros((height, int(height * 16 / 9), 3), dtype=np.uint8)
    else:
        width = max(1, round(image.shape[1] * height / image.shape[0]))
        tile = cv2.resize(image, (width, height))
    cv2.putText(tile, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return tile


def _bar(width, text, background, height=36, scale=0.7, text_color=(255, 255, 255)):
    bar = np.full((height, width, 3), background, dtype=np.uint8)
    cv2.putText(bar, text, (10, height - 11), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 2, cv2.LINE_AA)
    return bar


def compose_mosaic(images, status, state="idle", decision=None, tile_height=360):
    """Mosaico horizontal de las vistas con barra de estado y, si hay, el banner de decisión.

    images: {vista: imagen BGR o None}. decision: FinalDecision del último intento o None.
    """
    tiles = [_tile(images.get(view), VIEW_TITLES[view], tile_height) for view in VIEWS]
    mosaic = np.hstack(tiles)
    width = mosaic.shape[1]
    parts = [_bar(width, status, (40, 40, 40), text_color=STATE_COLORS.get(state, (255, 255, 255))), mosaic]
    if decision is not None:
        parts.append(_bar(width, f"{decision.display}: {decision.explanation}",
                          BANNER_COLORS[decision.decision], height=40, scale=0.6))
        parts.append(_bar(width, "   ".join(_vote_text(decision, view) for view in VIEWS), (40, 40, 40), height=32, scale=0.6))
    return np.vstack(parts)


def _vote_text(decision, view):
    verdict = decision.verdicts[view]
    names = {"valid": "VALIDO", "invalid": "NULO", "insufficient": "SIN DATOS"}
    text = f"{VIEW_TITLES[view]}: {names[verdict.label]}"
    if verdict.probability is not None:
        text += f" ({verdict.probability:.2f})"
    return text


def status_text(exercise, state, frames, controls, skipped=0):
    """Texto de la barra de estado con las teclas disponibles."""
    keys = f"[{controls.start_stop}] {'terminar' if state == 'recording' else 'iniciar'}  [{controls.cancel}] cancelar  [{controls.quit}] salir"
    if state == "recording":
        counts = "/".join(str(frames[v]) for v in VIEWS)
        return f"{exercise.upper()}  GRABANDO  frames F/I/D: {counts}  omitidos: {skipped}   {keys}"
    return f"{exercise.upper()}  en espera   {keys}"

