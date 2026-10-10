"""Extrae todos los keypoints de los modelos YOLO de peso muerto.

Uso: python extract_keypoints/dl/extract_keypoints_yolo_dl.py --view all
Salidas: keypoints/dl/yolo/{left,right}/dl_XXX.npy
Cada archivo tiene forma (T, K, 3), con K leido del modelo cargado:
x, y normalizados y confianza YOLO.
Los puntos se guardan en el orden original del modelo, sin recorte ni filtro
por confianza. Sin persona detectada, el frame completo queda en NaN.

DL frontal no tiene modelo YOLO pose (solo segmentacion); por eso aqui no se
contempla la vista front. El frontal de MediaPipe vive en su propio extractor.

Orden lateral confirmado en display/dl/yolo_skeleton_dl_side.py:
0 cadera, 1 hombro.
left:  YOLO 0,1 <-> MediaPipe 23,11 (cadera izq, hombro izq).
right: YOLO 0,1 <-> MediaPipe 24,12 (cadera der, hombro der).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "dl"
OUTPUT_ROOT = PROJECT_ROOT / "keypoints" / "dl" / "yolo"
MODEL_ROOT = PROJECT_ROOT / "models" / "dl"

CONFIG = {
    "left": "dl_side_skeleton_v1.pt",
    "right": "dl_side_skeleton_v1.pt",
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
                raise RuntimeError(f"Numero inesperado de resultados en {ruta}")
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
        raise RuntimeError(f"Modelo {nombre_modelo}: kpt_shape invalido: {forma}")
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
    parser.add_argument("--view", choices=("left", "right", "all"),
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
