"""Extrae landmarks MediaPipe de press de banca comparables con YOLO BP.

Uso: python extract_keypoints/bp/extract_keypoints_mediapipe_bp.py --view all
Salidas: keypoints/bp/mediapipe/{front,left,right}/bp_XXX.npy
Cada archivo tiene forma (T, K, 3): x, y normalizados y visibility MediaPipe.

front (K=6), en orden YOLO frontal: 13,11,12,14,16,15
(codo izq, hombro izq, hombro der, codo der, muneca der, muneca izq).

left (K=8), en orden YOLO lateral: 25,27,23,11,15,13,29,31
(rodilla izq, tobillo izq, cadera izq, hombro izq, muneca izq, codo izq,
talon izq, punta izq).
right (K=8), en orden YOLO lateral: 26,28,24,12,16,14,30,32
(rodilla der, tobillo der, cadera der, hombro der, muneca der, codo der,
talon der, punta der).

Visibility MediaPipe y confidence YOLO no son equivalentes.
"""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "bp"
OUTPUT_ROOT = PROJECT_ROOT / "keypoints" / "bp" / "mediapipe"
MODEL_PATH = PROJECT_ROOT / "models" / "common" / "pose_landmarker_heavy.task"

LANDMARKS = {
    "front": (13, 11, 12, 14, 16, 15),
    # Orden YOLO lateral confirmado: rodilla, tobillo, cadera, hombro,
    # muneca, codo, talon, punta (lado visible segun la vista).
    "left": (25, 27, 23, 11, 15, 13, 29, 31),
    "right": (26, 28, 24, 12, 16, 14, 30, 32),
}


def crear_landmarker():
    opciones = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(opciones)


def extraer_frame(resultado, indices):
    puntos = np.full((len(indices), 3), np.nan, dtype=np.float32)
    if not resultado.pose_landmarks:
        return puntos

    landmarks = resultado.pose_landmarks[0]
    if len(landmarks) <= max(indices):
        raise RuntimeError(f"MediaPipe devolvio {len(landmarks)} landmarks; "
                           f"se necesita el indice {max(indices)}")

    for posicion, indice in enumerate(indices):
        landmark = landmarks[indice]
        puntos[posicion] = (landmark.x, landmark.y, landmark.visibility)
    return puntos


def extraer_video(ruta, indices):
    captura = cv2.VideoCapture(str(ruta))
    if not captura.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {ruta}")

    fps = captura.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    secuencia = []
    landmarker = None
    try:
        landmarker = crear_landmarker()
        frame_idx = 0
        while True:
            valido, frame = captura.read()
            if not valido:
                break

            imagen_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen_rgb)
            timestamp_ms = max(frame_idx, int(round(frame_idx * 1000 / fps)))
            resultado = landmarker.detect_for_video(imagen_mp, timestamp_ms)
            secuencia.append(extraer_frame(resultado, indices))
            frame_idx += 1
    finally:
        captura.release()
        if landmarker is not None:
            landmarker.close()

    if not secuencia:
        return np.empty((0, len(indices), 3), dtype=np.float32)
    return np.stack(secuencia).astype(np.float32)


def procesar_vista(vista, overwrite):
    entrada = DATASET_ROOT / vista
    salida = OUTPUT_ROOT / vista
    videos = sorted(entrada.glob("*.mp4"))
    if not videos:
        raise RuntimeError(f"No se encontraron videos en: {entrada}")

    pendientes = [video for video in videos
                  if overwrite or not (salida / f"{video.stem}.npy").exists()]
    if not pendientes:
        print(f"{vista}: {len(videos)} archivos existentes; nada que procesar")
        return

    salida.mkdir(parents=True, exist_ok=True)
    print(f"{vista}: {len(pendientes)}/{len(videos)} videos | MediaPipe")
    for numero, video in enumerate(pendientes, 1):
        puntos = extraer_video(video, LANDMARKS[vista])
        archivo = salida / f"{video.stem}.npy"
        np.save(archivo, puntos)
        print(f"[{numero}/{len(pendientes)}] {archivo}: {puntos.shape}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=("front", "left", "right", "all"),
                        default="all", help="Vista a extraer (por defecto: all)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reemplazar .npy MediaPipe ya existentes")
    argumentos = parser.parse_args()
    vistas = LANDMARKS if argumentos.view == "all" else (argumentos.view,)
    for vista in vistas:
        procesar_vista(vista, argumentos.overwrite)


if __name__ == "__main__":
    main()
