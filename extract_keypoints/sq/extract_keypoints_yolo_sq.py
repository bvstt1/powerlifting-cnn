"""Extrae todos los keypoints de los modelos YOLO de sentadilla.

Uso: python extract_keypoints/sq/extract_keypoints_yolo_sq.py --view all
Salidas: keypoints/sq/yolo/{front,left,right}/sq_XXX.npy
Cada archivo tiene forma (T, K, 3), con K leído del modelo cargado:
x, y normalizados y confianza YOLO.
Los puntos se guardan en el orden original del modelo, sin recorte ni filtro
por confianza. Sin persona detectada, el frame completo queda en NaN.

Correspondencias aproximadas con MediaPipe (no alteran la salida YOLO):
front (16 puntos según el esqueleto de visualización, índices YOLO 0..15):
    hombros  YOLO: 0,1   <-> MediaPipe: 11,12
    codos    YOLO: 12,13 <-> MediaPipe: 13,14
    muñecas  YOLO: 15,14 <-> MediaPipe: 15,16
    caderas  YOLO: 3,2   <-> MediaPipe: 23,24
    rodillas YOLO: 4,5   <-> MediaPipe: 25,26
    tobillos YOLO: 9,6   <-> MediaPipe: 27,28
    pie der. YOLO: 7 puntas, 8 talones <-> MediaPipe: 32,30
    pie izq. YOLO: 10 puntas, 11 talones <-> MediaPipe: 31,29
    La identidad punta/talón de esos índices fue confirmada manualmente.
    El lado izquierdo/derecho frontal se deduce de las conexiones del
    esqueleto en display/sq/fast/yolo_skeleton_sq_front.py; confirmar con la
    anotación de entrenamiento antes de comparar lados anatómicos.
left (K=5):  YOLO 0,1,2,3,4 <-> MediaPipe 25,23,27,31,29
right (K=5): YOLO 0,1,2,3,4 <-> MediaPipe 26,24,28,32,30
    Orden lateral confirmado: rodilla, cadera, tobillo, punta, talón.
    El orden lateral es independiente del frontal.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "sq"
OUTPUT_ROOT = PROJECT_ROOT / "keypoints" / "sq" / "yolo"
MODEL_ROOT = PROJECT_ROOT / "models" / "sq"

# Los esqueletos de visualización muestran 16 puntos front y 5 side;
# el número real se toma del checkpoint sin eliminar ni reordenar puntos.
CONFIG = {
    "front": "sq_front_skeleton_v5.pt",
    "left": "sq_side_skeleton_v3.pt",
    "right": "sq_side_skeleton_v3.pt",
}
IMG_SIZE = 960
BATCH_SIZE = 16
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def extraer_frame(resultado, total):
    puntos = np.full((total, 3), np.nan, dtype=np.float32)
    if resultado.keypoints is None or len(resultado.keypoints) == 0:
        return puntos

    if resultado.boxes is not None and len(resultado.boxes) > 0:
        persona = int(np.argmax(resultado.boxes.conf.cpu().numpy()))
    else:
        persona = 0

    coordenadas = resultado.keypoints.xyn[persona].cpu().numpy().astype(np.float32)
    if coordenadas.shape != (total, 2):
        raise RuntimeError(f"Se esperaban {total} puntos, recibidos: {coordenadas.shape}")

    if resultado.keypoints.conf is None:
        confianza = np.ones(total, dtype=np.float32)
    else:
        confianza = resultado.keypoints.conf[persona].cpu().numpy().astype(np.float32)
    if confianza.shape != (total,):
        raise RuntimeError(f"Se esperaban {total} confianzas, recibidas: {confianza.shape}")

    puntos[:, :2] = coordenadas
    puntos[:, 2] = confianza
    return puntos


def extraer_video(ruta, modelo, total):
    captura = cv2.VideoCapture(str(ruta))
    if not captura.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {ruta}")

    secuencia = []
    try:
        while True:
            frames = []
            for _ in range(BATCH_SIZE):
                valido, frame = captura.read()
                if not valido:
                    break
                frames.append(frame)
            if not frames:
                break

            resultados = modelo(frames, imgsz=IMG_SIZE, device=DEVICE, verbose=False)
            if len(resultados) != len(frames):
                raise RuntimeError(f"Número inesperado de resultados en {ruta}")
            secuencia.extend(
                extraer_frame(resultado, total)
                for resultado in resultados
            )
    finally:
        captura.release()

    if not secuencia:
        return np.empty((0, total, 3), dtype=np.float32)
    return np.stack(secuencia).astype(np.float32)


def procesar_vista(vista, modelos, overwrite):
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

    nombre_modelo = CONFIG[vista]
    if nombre_modelo not in modelos:
        modelos[nombre_modelo] = YOLO(str(MODEL_ROOT / nombre_modelo))
    modelo = modelos[nombre_modelo]
    forma = modelo.model.yaml.get("kpt_shape", [])
    if forma is None or len(forma) != 2 or forma[0] <= 0 or forma[1] != 3:
        raise RuntimeError(f"Modelo {nombre_modelo}: kpt_shape inválido: {forma}")
    total = int(forma[0])

    salida.mkdir(parents=True, exist_ok=True)
    print(f"{vista}: {len(pendientes)}/{len(videos)} videos | "
          f"{nombre_modelo} ({total} puntos)")
    for numero, video in enumerate(pendientes, 1):
        puntos = extraer_video(video, modelo, total)
        archivo = salida / f"{video.stem}.npy"
        np.save(archivo, puntos)
        print(f"[{numero}/{len(pendientes)}] {archivo}: {puntos.shape}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=("front", "left", "right", "all"),
                        default="all", help="Vista a extraer (por defecto: all)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reemplazar .npy YOLO ya existentes")
    argumentos = parser.parse_args()
    vistas = CONFIG if argumentos.view == "all" else (argumentos.view,)
    modelos = {}
    for vista in vistas:
        procesar_vista(vista, modelos, argumentos.overwrite)


if __name__ == "__main__":
    main()
