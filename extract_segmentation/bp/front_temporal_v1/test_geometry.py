"""Prueba reglas geométricas front con máscaras y poses sintéticas.

Comprueba guardia inferior, duplicados e inclinación sin YOLO ni etiquetas.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import geometry as G


def _crop_rect(h, w):
    return np.ones((h, w), dtype=bool)


def test_guardia_inferior():
    b = {"idx_global": 0, "clase": "bar", "conf": 0.9,
         "bbox": np.array([900, 700, 1100, 740]),
         "track_id": 0, "mascara": _crop_rect(40, 200),
         "origen": np.array([900, 700]), "area": 8000}
    p = {"idx_global": 1, "clase": "person", "conf": 0.9,
         "bbox": np.array([700, 100, 1300, 900]),
         "track_id": 2, "mascara": _crop_rect(10, 10),
         "origen": np.array([700, 100]), "area": 100}
    pose = np.full((6, 3), np.nan, np.float32)
    # munecas altas (y=0.3), barra abajo (y~720/1080=0.67)
    pose[5] = [0.4, 0.3, 0.99]
    pose[4] = [0.6, 0.3, 0.99]
    pose[1] = [0.42, 0.5, 0.99]
    pose[2] = [0.58, 0.5, 0.99]
    sel = G.seleccionar_barras([b, p], pose, 1920, 1080)
    assert sel["estado"] == "observada_no_verificada", sel
    print("OK guardia_inferior")


def test_dedup():
    m = _crop_rect(40, 200)
    b1 = {"idx_global": 0, "clase": "bar", "conf": 0.9,
          "bbox": np.array([800, 400, 1100, 440]), "track_id": 0,
          "mascara": m, "origen": np.array([800, 400]), "area": 8000}
    b2 = {"idx_global": 1, "clase": "bar", "conf": 0.8,
          "bbox": np.array([800, 400, 1100, 440]), "track_id": 0,
          "mascara": m.copy(), "origen": np.array([800, 400]), "area": 8000}
    p = {"idx_global": 2, "clase": "person", "conf": 0.9,
         "bbox": np.array([600, 100, 1300, 900]), "track_id": 2,
         "mascara": _crop_rect(10, 10), "origen": np.array([600, 100]),
         "area": 100}
    pose = np.full((6, 3), np.nan, np.float32)
    pose[5] = [0.35, 0.4, 0.99]
    pose[4] = [0.65, 0.4, 0.99]
    pose[1] = [0.42, 0.5, 0.99]
    pose[2] = [0.58, 0.5, 0.99]
    sel = G.seleccionar_barras([b1, b2, p], pose, 1920, 1080)
    assert sel["n_kept"] == 1, sel
    print("OK dedup")


def test_inclinacion_rechazo():
    # mascara vertical -> pendiente alta -> None
    m = np.zeros((200, 40), dtype=bool)
    m[:, 18:22] = True
    eleg = [{"mascara": m, "origen": np.array([500, 300])}]
    assert G.inclinacion_barra_union(eleg, 1080) is None
    print("OK inclinacion_rechazo")


if __name__ == "__main__":
    test_guardia_inferior()
    test_dedup()
    test_inclinacion_rechazo()
    print("todas las pruebas OK")
