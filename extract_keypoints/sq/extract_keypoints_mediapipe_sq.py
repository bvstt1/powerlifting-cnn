"""Extrae landmarks MediaPipe de sentadilla comparables con YOLO SQ.

Uso: python extract_keypoints/sq/extract_keypoints_mediapipe_sq.py --view all
Salidas: keypoints/sq/mediapipe/{front,left,right}/sq_XXX.npy
Cada archivo tiene forma (T, K, 3): x, y normalizados y visibility MediaPipe.

Posición de salida -> índice MediaPipe (primeros puntos en orden YOLO):
front (K=16):
    0->11 hombro izq.; 1->12 hombro der.; 2->24 cadera der.; 3->23 cadera izq.
    4->25 rodilla izq.; 5->26 rodilla der.; 6->28 tobillo der.
    7->32 punta der.; 8->30 talón der.; 9->27 tobillo izq.
    10->31 punta izq.; 11->29 talón izq.; 12->13 codo izq.
    13->14 codo der.; 14->16 muñeca der.; 15->15 muñeca izq.
left (K=5):  25,23,27,31,29 (rodilla, cadera, tobillo, punta, talón).
right (K=5): 26,24,28,32,30 (mismo orden, lado derecho).
En lateral cada posición coincide con YOLO 0..4; su orden es independiente
del modelo frontal y no incluye el hombro.

Los lados del modelo YOLO frontal se dedujeron del esqueleto de visualización;
confirmar sus etiquetas anatómicas antes de comparar izquierda/derecha.
Las etiquetas talón/punta YOLO front (8,11 / 7,10) fueron confirmadas
manualmente. Visibility MediaPipe y confidence YOLO no son equivalentes.
"""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "sq"
OUTPUT_ROOT = PROJECT_ROOT / "keypoints" / "sq" / "mediapipe"
MODEL_PATH = PROJECT_ROOT / "models" / "common" / "pose_landmarker_heavy.task"

# Front coincide con YOLO front 0..15; lateral con YOLO side 0..4.
LANDMARKS = {
    "front": (11, 12, 24, 23, 25, 26, 28, 32, 30, 27, 31, 29, 13, 14, 16, 15),
    "left": (25, 23, 27, 31, 29),
    "right": (26, 24, 28, 32, 30),
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
        raise RuntimeError(f"MediaPipe devolvió {len(landmarks)} landmarks; "
                           f"se necesita el índice {max(indices)}")

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
        # Un landmarker nuevo por video evita arrastrar el estado temporal.
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
