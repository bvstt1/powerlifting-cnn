"""Visualiza YOLO DL lateral con el formato del visor de sentadilla.

Configura VISTA, MODEL_PATH y VIDEO_PATH debajo y ejecuta sin argumentos.
Tambien admite --view left|right, --model y --video para cambiar de vista,
checkpoint o video sin editar el archivo.
ESPACIO: pausa/reanuda | N: siguiente frame en pausa | Q/Esc: salir.

DL frontal no tiene modelo YOLO pose; este visor solo cubre left/right.
Orden lateral: 0 cadera, 1 hombro.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "models" / "dl"

# CONFIG: cambia la vista, el modelo o el video aqui.
VISTA = "right"  # "left" o "right"
MODELOS = {
    "left": "dl_side_skeleton_v1.pt",
    "right": "dl_side_skeleton_v1.pt",
}
MODEL_PATH = MODEL_ROOT / MODELOS[VISTA]
VIDEO_PATH = PROJECT_ROOT / "dataset" / "dl" / VISTA / "dl_373.mp4"

NOMBRES = {
    "left": ("cadera izq", "hombro izq"),
    "right": ("cadera der", "hombro der"),
}

# Landmarks de MediaPipe equivalentes, solo para mostrarlos en el panel.
INDICES_MP = {
    "left": (23, 11),
    "right": (24, 12),
}
CONEXIONES = {
    "left": ((1, 0),),
    "right": ((1, 0),),
}
IMG_SIZE = 960
MIN_CONF = 0.5
ALTURA_VISTA = 800
ANCHO_PANEL = 420
ANCHO_VENTANA = 1500
ALTO_VENTANA = 850


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


def ruta_modelo(nombre):
    ruta = Path(nombre)
    if not ruta.is_file():
        ruta = MODEL_ROOT / nombre
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontro el modelo: {ruta}")
    return ruta


def puntos_del_modelo(resultado, total):
    puntos = np.full((total, 3), np.nan, dtype=np.float32)
    if resultado.keypoints is None or len(resultado.keypoints) == 0:
        return puntos

    if resultado.boxes is not None and len(resultado.boxes) > 0:
        persona = int(np.argmax(resultado.boxes.conf.cpu().numpy()))
    else:
        persona = 0
    xy = resultado.keypoints.xyn[persona].cpu().numpy()
    confianza = resultado.keypoints.conf
    confianza = (confianza[persona].cpu().numpy()
                 if confianza is not None else np.ones(total, dtype=np.float32))
    if xy.shape != (total, 2) or confianza.shape != (total,):
        raise RuntimeError(f"Puntos inesperados: xy={xy.shape}, confianza={confianza.shape}")
    puntos[:, :2] = xy
    puntos[:, 2] = confianza
    return puntos


def dibujar(frame, puntos, vista, numero, nombre_modelo):
    alto_original, ancho_original = frame.shape[:2]
    ancho = int(round(ancho_original * ALTURA_VISTA / alto_original))
    vista_video = cv2.resize(frame, (ancho, ALTURA_VISTA))
    panel = np.full((ALTURA_VISTA, ANCHO_PANEL, 3), 28, dtype=np.uint8)

    posiciones = []
    for x, y, confianza in puntos:
        if (not np.isfinite([x, y, confianza]).all()
                or (x == 0 and y == 0) or not (0 <= x <= 1 and 0 <= y <= 1)):
            posiciones.append(None)
        else:
            posiciones.append((int(x * (ancho - 1)), int(y * (ALTURA_VISTA - 1))))
    validos = np.isfinite(puntos[:, 2]) & (puntos[:, 2] >= MIN_CONF)

    for a, b in CONEXIONES[vista]:
        if posiciones[a] is not None and posiciones[b] is not None and validos[a] and validos[b]:
            cv2.line(vista_video, posiciones[a], posiciones[b], (0, 180, 0), 2)

    cv2.putText(panel, f"DL {vista} | YOLO directo", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(panel, nombre_modelo, (12, 61),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.putText(panel, f"Frame: {numero}  |  Puntos: {len(puntos)}", (12, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(panel, "Etiqueta: YOLO / MP / articulacion / conf", (12, 114),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1)

    for indice, (mp_idx, nombre) in enumerate(zip(INDICES_MP[vista], NOMBRES[vista])):
        color = (0, 230, 230) if validos[indice] else (130, 130, 130)
        posicion = posiciones[indice]
        if posicion is not None:
            cv2.circle(vista_video, posicion, 5, color, -1)
            cv2.putText(vista_video, f"{indice}/{mp_idx}",
                        (posicion[0] + 6, posicion[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

        confianza = puntos[indice, 2]
        conf_texto = f"{confianza:.2f}" if np.isfinite(confianza) else "--"
        cv2.putText(panel, f"{indice:02d} / {mp_idx:02d} / {nombre} / {conf_texto}",
                    (12, 146 + indice * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    color, 1)

    return np.hstack((vista_video, panel))


def visualizar(vista, video, modelo_path):
    modelo = YOLO(str(modelo_path))
    forma = modelo.model.yaml.get("kpt_shape", [])
    if list(forma or []) != [len(NOMBRES[vista]), 3]:
        raise ValueError(f"{modelo_path.name} tiene kpt_shape={forma}; "
                         f"la vista {vista} requiere {[len(NOMBRES[vista]), 3]}")

    captura = cv2.VideoCapture(str(video))
    if not captura.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video}")

    ventana = f"DL YOLO - {vista}"
    try:
        fps = captura.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        delay = max(1, int(round(1000 / fps)))
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
                resultado = modelo(frame, imgsz=IMG_SIZE, verbose=False)[0]
                puntos = puntos_del_modelo(resultado, len(NOMBRES[vista]))
                cv2.imshow(ventana, dibujar(frame, puntos, vista, numero, modelo_path.name))
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
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=MODELOS, help="Vista alternativa a VISTA")
    parser.add_argument("--model", help="Ruta o nombre del .pt alternativo a MODEL_PATH")
    parser.add_argument("--video", help="Ruta del mp4 o ID alternativo a VIDEO_PATH")
    args = parser.parse_args()
    vista = args.view or VISTA
    modelo = (args.model if args.model else
              MODEL_PATH if args.view is None else MODEL_ROOT / MODELOS[vista])
    video = (ruta_video(vista, str(VIDEO_PATH)) if args.view is None and args.video is None
             else ruta_video(vista, args.video))
    visualizar(vista, video, ruta_modelo(modelo))


if __name__ == "__main__":
    main()
