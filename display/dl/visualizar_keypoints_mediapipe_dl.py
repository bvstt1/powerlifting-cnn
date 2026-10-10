"""Visualiza directamente los landmarks del modelo MediaPipe en peso muerto.

Ejemplos:
    python display/dl/visualizar_keypoints_mediapipe_dl.py
    python display/dl/visualizar_keypoints_mediapipe_dl.py --view front --video dl_010
    python display/dl/visualizar_keypoints_mediapipe_dl.py --view left --video dl_004
    python display/dl/visualizar_keypoints_mediapipe_dl.py --view right --video dl_004

El modelo detecta la pose de cada frame; no usa archivos .npy ni extractores.
front muestra cuerpo sin rostro + centroides de manos y barra simulada;
left/right muestran cadera y hombro del lado correspondiente en orden YOLO.
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
VISTA = "right"  # "front", "left" o "right"
VIDEO_PATH = PROJECT_ROOT / "dataset" / "dl" / VISTA / "dl_373.mp4"

MODEL_PATH = PROJECT_ROOT / "models" / "common" / "pose_landmarker_heavy.task"

CUERPO = list(range(11, 33))
MANO_IZQ = [15, 17, 19, 21]
MANO_DER = [16, 18, 20, 22]
LANDMARKS_SIDE = {
    "left": (23, 11),
    "right": (24, 12),
}
NOMBRES_SIDE = {
    "left": ("cadera izq", "hombro izq"),
    "right": ("cadera der", "hombro der"),
}
CONEXIONES_CUERPO = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]
MIN_VISIBILIDAD = 0.5
OFFSET_Y_PX = 19
ALTURA_VISTA = 800
ANCHO_PANEL = 420
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
    carpeta = PROJECT_ROOT / "dataset" / "dl" / vista
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


def centro_mano(landmarks, indices, ancho, alto):
    xs, ys = [], []
    for idx in indices:
        lm = landmarks[idx]
        if np.isfinite([lm.x, lm.y, lm.visibility]).all() and lm.visibility >= MIN_VISIBILIDAD:
            xs.append(lm.x * (ancho - 1))
            ys.append(lm.y * (alto - 1))
    if not xs:
        return None
    return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)) + OFFSET_Y_PX)


def dibujar(frame, landmarks, vista, numero):
    alto_original, ancho_original = frame.shape[:2]
    ancho = int(round(ancho_original * ALTURA_VISTA / alto_original))
    vista_video = cv2.resize(frame, (ancho, ALTURA_VISTA))
    panel = np.full((ALTURA_VISTA, ANCHO_PANEL, 3), 28, dtype=np.uint8)

    cv2.putText(panel, f"DL {vista} | MediaPipe directo", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(panel, f"Frame: {numero}", (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    if landmarks is None:
        cv2.putText(panel, "Sin pose detectada", (12, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        return np.hstack((vista_video, panel))

    if vista == "front":
        posiciones = {}
        for idx in CUERPO:
            lm = landmarks[idx]
            if (np.isfinite([lm.x, lm.y, lm.visibility]).all()
                    and lm.visibility >= MIN_VISIBILIDAD
                    and 0 <= lm.x <= 1 and 0 <= lm.y <= 1):
                pos = (int(lm.x * (ancho - 1)), int(lm.y * (ALTURA_VISTA - 1)))
                posiciones[idx] = pos
                radio = 9 if idx in MANO_IZQ + MANO_DER else 4
                color = (0, 255, 255) if idx in MANO_IZQ + MANO_DER else (0, 230, 230)
                cv2.circle(vista_video, pos, radio, color, -1)

        for a, b in CONEXIONES_CUERPO:
            if a in posiciones and b in posiciones:
                cv2.line(vista_video, posiciones[a], posiciones[b], (0, 180, 0), 2)

        izq = centro_mano(landmarks, MANO_IZQ, ancho, ALTURA_VISTA)
        der = centro_mano(landmarks, MANO_DER, ancho, ALTURA_VISTA)
        if izq is not None:
            cv2.circle(vista_video, izq, 8, (0, 255, 255), -1)
        if der is not None:
            cv2.circle(vista_video, der, 8, (0, 255, 255), -1)
        if izq is not None and der is not None:
            cv2.line(vista_video, izq, der, (0, 255, 255), 3)
        cv2.putText(panel, "Cuerpo 11..32 + centroides manos + barra", (12, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1)
    else:
        cv2.putText(panel, "Etiqueta: YOLO / MP / articulacion / vis", (12, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1)
        for indice, (mp_idx, nombre) in enumerate(zip(LANDMARKS_SIDE[vista], NOMBRES_SIDE[vista])):
            lm = landmarks[mp_idx]
            valido = (np.isfinite([lm.x, lm.y, lm.visibility]).all()
                      and lm.visibility >= MIN_VISIBILIDAD
                      and 0 <= lm.x <= 1 and 0 <= lm.y <= 1)
            color = (0, 230, 230) if valido else (130, 130, 130)
            if valido:
                pos = (int(lm.x * (ancho - 1)), int(lm.y * (ALTURA_VISTA - 1)))
                cv2.circle(vista_video, pos, 5, color, -1)
                cv2.putText(vista_video, f"{indice}/{mp_idx}",
                            (pos[0] + 6, pos[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
            vis_texto = f"{lm.visibility:.2f}" if np.isfinite(lm.visibility) else "--"
            cv2.putText(panel, f"{indice} / {mp_idx:02d} / {nombre} / {vis_texto}",
                        (12, 122 + indice * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        color, 1)
        # Conexion hombro-cadera si ambos son visibles.
        lm_cadera = landmarks[LANDMARKS_SIDE[vista][0]]
        lm_hombro = landmarks[LANDMARKS_SIDE[vista][1]]
        if (lm_cadera.visibility >= MIN_VISIBILIDAD
                and lm_hombro.visibility >= MIN_VISIBILIDAD):
            p0 = (int(lm_cadera.x * (ancho - 1)), int(lm_cadera.y * (ALTURA_VISTA - 1)))
            p1 = (int(lm_hombro.x * (ancho - 1)), int(lm_hombro.y * (ALTURA_VISTA - 1)))
            cv2.line(vista_video, p0, p1, (0, 180, 0), 2)

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

        ventana = f"DL MediaPipe - {vista}"
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
    parser.add_argument("--view", choices=("front", "left", "right"), help="Vista alternativa a VISTA")
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
