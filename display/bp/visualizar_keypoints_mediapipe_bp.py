"""Visualiza directamente los landmarks del modelo MediaPipe en press de banca.

Ejemplos:
    python display/bp/visualizar_keypoints_mediapipe_bp.py
    python display/bp/visualizar_keypoints_mediapipe_bp.py --view front --video bp_622
    python display/bp/visualizar_keypoints_mediapipe_bp.py --view left --video bp_400
    python display/bp/visualizar_keypoints_mediapipe_bp.py --view right --video bp_400

El modelo detecta la pose de cada frame; no usa archivos .npy ni extractores.
front muestra tren superior en orden YOLO; left/right muestran los 8 puntos
en el orden YOLO lateral confirmado.
ESPACIO: pausa/reanuda | N: siguiente frame en pausa | Q/Esc: salir.
"""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# CONFIG: elige aqui la vista y el video.
VISTA = "left"  # "front", "left" o "right"
VIDEO_PATH = PROJECT_ROOT / "dataset" / "bp" / VISTA / "bp_001.mp4"

MODEL_PATH = PROJECT_ROOT / "models" / "common" / "pose_landmarker_heavy.task"

# front en orden YOLO frontal; laterales en orden YOLO lateral confirmado.
LANDMARKS = {
    "front": (13, 11, 12, 14, 16, 15),
    "left": (25, 27, 23, 11, 15, 13, 29, 31),
    "right": (26, 28, 24, 12, 16, 14, 30, 32),
}
NOMBRES = {
    "front": ("codo izq", "hombro izq", "hombro der",
              "codo der", "muneca der", "muneca izq"),
    "left": ("rodilla izq", "tobillo izq", "cadera izq", "hombro izq",
             "muneca izq", "codo izq", "talon izq", "punta izq"),
    "right": ("rodilla der", "tobillo der", "cadera der", "hombro der",
              "muneca der", "codo der", "talon der", "punta der"),
}
CONEXIONES = {
    "front": ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5)),
    "left": ((2, 0), (0, 1), (1, 6), (6, 7), (1, 7),
             (2, 3), (3, 5), (5, 4)),
    "right": ((2, 0), (0, 1), (1, 6), (6, 7), (1, 7),
              (2, 3), (3, 5), (5, 4)),
}
MIN_VISIBILIDAD = 0.5
ALTURA_VISTA = 800
ANCHO_PANEL = 460
ANCHO_VENTANA = 1500
ALTO_VENTANA = 850


def crear_landmarker():
    opciones = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(opciones)


def ruta_video(vista, nombre):
    carpeta = PROJECT_ROOT / "dataset" / "bp" / vista
    if nombre is None:
        videos = sorted(carpeta.glob("*.mp4"))
        if not videos:
            raise FileNotFoundError(f"No hay videos en {carpeta}")
        return videos[0]

    ruta = Path(nombre)
    if not ruta.is_file():
        ruta = carpeta / (nombre if ruta.suffix else f"{nombre}.mp4")
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontro el video: {ruta}")
    return ruta


def coordenadas_visibles(landmarks, vista, ancho, alto):
    coordenadas = []
    visibilidades = []
    for indice_mp in LANDMARKS[vista]:
        if landmarks is None:
            coordenadas.append(None)
            visibilidades.append(float("nan"))
            continue
        punto = landmarks[indice_mp]
        x, y, visibilidad = punto.x, punto.y, punto.visibility
        visibilidades.append(visibilidad)
        if not np.isfinite([x, y, visibilidad]).all() or not (0 <= x <= 1 and 0 <= y <= 1):
            coordenadas.append(None)
        else:
            coordenadas.append((int(x * (ancho - 1)), int(y * (alto - 1))))
    return coordenadas, visibilidades


def dibujar(frame, landmarks, vista, numero):
    alto_original, ancho_original = frame.shape[:2]
    ancho = int(round(ancho_original * ALTURA_VISTA / alto_original))
    vista_video = cv2.resize(frame, (ancho, ALTURA_VISTA))
    panel = np.full((ALTURA_VISTA, ANCHO_PANEL, 3), 28, dtype=np.uint8)
    posiciones, visibilidades = coordenadas_visibles(landmarks, vista, ancho, alto=ALTURA_VISTA)
    validos = [np.isfinite(vis) and vis >= MIN_VISIBILIDAD for vis in visibilidades]

    for a, b in CONEXIONES[vista]:
        if posiciones[a] is not None and posiciones[b] is not None and validos[a] and validos[b]:
            cv2.line(vista_video, posiciones[a], posiciones[b], (0, 180, 0), 2)

    cv2.putText(panel, f"BP {vista} | MediaPipe directo", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(panel, f"Frame: {numero}  |  Puntos: {len(LANDMARKS[vista])}", (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(panel, "Etiqueta: YOLO / MP / articulacion / vis", (12, 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1)

    for indice, (mp_idx, nombre) in enumerate(zip(LANDMARKS[vista], NOMBRES[vista])):
        color = (0, 230, 230) if validos[indice] else (130, 130, 130)
        posicion = posiciones[indice]
        if posicion is not None:
            cv2.circle(vista_video, posicion, 5, color, -1)
            cv2.putText(vista_video, f"{indice}/{mp_idx}",
                        (posicion[0] + 6, posicion[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

        visibilidad = visibilidades[indice]
        vis_texto = f"{visibilidad:.2f}" if np.isfinite(visibilidad) else "--"
        cv2.putText(panel, f"{indice:02d} / {mp_idx:02d} / {nombre} / {vis_texto}",
                    (12, 118 + indice * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    color, 1)

    return np.hstack((vista_video, panel))


def visualizar(vista, video):
    captura = cv2.VideoCapture(str(video))
    if not captura.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video}")

    landmarker = None
    try:
        fps = captura.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        delay = max(1, int(round(1000 / fps)))
        landmarker = crear_landmarker()

        ventana = f"BP MediaPipe - {vista}"
        cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(ventana, ANCHO_VENTANA, ALTO_VENTANA)
        pausado = False
        avanzar = False
        numero = 0
        while True:
            if not pausado or avanzar:
                ok, frame = captura.read()
                if not ok:
                    break
                imagen_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen_rgb)
                timestamp = max(numero, int(round(numero * 1000 / fps)))
                resultado = landmarker.detect_for_video(imagen_mp, timestamp)
                landmarks = resultado.pose_landmarks[0] if resultado.pose_landmarks else None
                cv2.imshow(ventana, dibujar(frame, landmarks, vista, numero))
                numero += 1
                avanzar = False

            tecla = cv2.waitKey(0 if pausado else delay) & 0xFF
            if tecla in (27, ord("q")):
                break
            if tecla == ord(" "):
                pausado = not pausado
            if tecla == ord("n") and pausado:
                avanzar = True
    finally:
        captura.release()
        if landmarker is not None:
            landmarker.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=LANDMARKS, help="Vista alternativa a VISTA")
    parser.add_argument("--video", help="Ruta del mp4 o ID alternativo a VIDEO_PATH")
    args = parser.parse_args()
    vista = args.view or VISTA
    if args.video is None and args.view is None:
        video = ruta_video(vista, str(VIDEO_PATH))
    else:
        video = ruta_video(vista, args.video)
    visualizar(vista, video)


if __name__ == "__main__":
    main()
