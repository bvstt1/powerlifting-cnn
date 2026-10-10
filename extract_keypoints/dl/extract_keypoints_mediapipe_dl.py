"""Extrae landmarks MediaPipe de peso muerto comparables con YOLO DL.

Uso: python extract_keypoints/dl/extract_keypoints_mediapipe_dl.py --view all
Salidas: keypoints/dl/mediapipe/{front,left,right}/dl_XXX.npy

front (K=25): mantiene el formato vigente del extractor frontal:
22 landmarks corporales sin rostro (11..32) + centroide mano izq + centroide
mano der + distancia horizontal barra-rodilla. Cada fila es (x, y, visibility).
No tiene equivalente YOLO porque DL frontal no usa modelo YOLO pose.

left (K=2):  23,11 (cadera izq, hombro izq), en orden YOLO 0,1.
right (K=2): 24,12 (cadera der, hombro der), en orden YOLO 0,1.

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
DATASET_ROOT = PROJECT_ROOT / "dataset" / "dl"
OUTPUT_ROOT = PROJECT_ROOT / "keypoints" / "dl" / "mediapipe"
MODEL_PATH = PROJECT_ROOT / "models" / "common" / "pose_landmarker_heavy.task"

FACE_IDXS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
MANO_IZQ = [15, 17, 19, 21]
MANO_DER = [16, 18, 20, 22]
RODILLA_IZQ = 25
RODILLA_DER = 26
OFFSET_Y_NORM = 19 / 720.0

# Laterales en orden YOLO (0 cadera, 1 hombro).
LANDMARKS_SIDE = {
    "left": (23, 11),
    "right": (24, 12),
}


def crear_landmarker():
    opciones = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(opciones)


def extraer_frame_side(resultado, indices):
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


def extraer_frame_front(landmarks):
    # 22 landmarks del cuerpo sin rostro.
    frame_kps = [[lm.x, lm.y, lm.visibility]
                 for idx, lm in enumerate(landmarks) if idx not in FACE_IDXS]

    pts_izq = [(landmarks[i].x, landmarks[i].y)
               for i in MANO_IZQ if landmarks[i].visibility >= 0.5]
    if pts_izq:
        cx_i = float(sum(p[0] for p in pts_izq) / len(pts_izq))
        cy_i = float(sum(p[1] for p in pts_izq) / len(pts_izq)) + OFFSET_Y_NORM
        frame_kps.append([cx_i, cy_i, 1.0])
    else:
        frame_kps.append([np.nan, np.nan, 0.0])

    pts_der = [(landmarks[i].x, landmarks[i].y)
               for i in MANO_DER if landmarks[i].visibility >= 0.5]
    if pts_der:
        cx_d = float(sum(p[0] for p in pts_der) / len(pts_der))
        cy_d = float(sum(p[1] for p in pts_der) / len(pts_der)) + OFFSET_Y_NORM
        frame_kps.append([cx_d, cy_d, 1.0])
    else:
        frame_kps.append([np.nan, np.nan, 0.0])

    if (pts_izq and pts_der
            and landmarks[RODILLA_IZQ].visibility >= 0.5
            and landmarks[RODILLA_DER].visibility >= 0.5):
        bar_x = (cx_i + cx_d) / 2.0
        frame_kps.append([float(bar_x - landmarks[RODILLA_IZQ].x),
                          float(bar_x - landmarks[RODILLA_DER].x), 1.0])
    else:
        frame_kps.append([np.nan, np.nan, 0.0])

    return np.array(frame_kps, dtype=np.float32)


def extraer_video(ruta, vista):
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

            if vista == "front":
                if resultado.pose_landmarks:
                    secuencia.append(extraer_frame_front(resultado.pose_landmarks[0]))
                else:
                    secuencia.append(np.full((25, 3), np.nan, dtype=np.float32))
            else:
                secuencia.append(extraer_frame_side(resultado, LANDMARKS_SIDE[vista]))
            frame_idx += 1
    finally:
        captura.release()
        if landmarker is not None:
            landmarker.close()

    if not secuencia:
        k = 25 if vista == "front" else len(LANDMARKS_SIDE[vista])
        return np.empty((0, k, 3), dtype=np.float32)
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
        puntos = extraer_video(video, vista)
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
    vistas = ("front", "left", "right") if argumentos.view == "all" else (argumentos.view,)
    for vista in vistas:
        procesar_vista(vista, argumentos.overwrite)


if __name__ == "__main__":
    main()
