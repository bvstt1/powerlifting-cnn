"""Visualiza YOLO SQ frontal y lateral con el formato del visor MediaPipe.

Configura VISTA, MODEL_PATH y VIDEO_PATH debajo y ejecuta sin argumentos.
También admite --view front|left|right, --model y --video para cambiar de
vista, checkpoint o vídeo sin editar el archivo.
ESPACIO: pausa/reanuda | N: siguiente frame en pausa | Q/Esc: salir.
Tracking opcional: T activa | S pausa/selecciona con el mouse | R reinicia
| B retrocede un frame | ,/. saltan 30 frames | [/] cambia vídeo | E exporta CSV
| M alterna curva EMA independiente de la trayectoria original.
"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from .bar_tracker import BarTracker, VideoViewport, draw_vertical_plot
except ImportError:  # ejecución directa: python display/sq/visualizar_keypoints_yolo_sq.py
    from bar_tracker import BarTracker, VideoViewport, draw_vertical_plot


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "models" / "sq"

# CONFIG: cambia la vista, el modelo o el vídeo aquí.
VISTA = "left"  # "front", "left" o "right"
MODELOS = {
    "front": "sq_front_skeleton_v5.pt",
    "left": "sq_side_skeleton_v3.pt",
    "right": "sq_side_skeleton_v3.pt",
}
MODEL_PATH = MODEL_ROOT / MODELOS[VISTA]
VIDEO_PATH = PROJECT_ROOT / "dataset" / "sq" / VISTA / "sq_101.mp4"

# El frontal y el lateral tienen órdenes distintos; lateral confirmado:
# 0 rodilla, 1 cadera, 2 tobillo, 3 punta, 4 talón.
NOMBRES = {
    "front": (
        "hombro izq", "hombro der", "cadera der", "cadera izq",
        "rodilla izq", "rodilla der", "tobillo der", "punta der",
        "talon der", "tobillo izq", "punta izq", "talon izq",
        "codo izq", "codo der", "muneca der", "muneca izq",
    ),
    "left": ("rodilla izq", "cadera izq", "tobillo izq",
             "punta izq", "talon izq"),
    "right": ("rodilla der", "cadera der", "tobillo der",
              "punta der", "talon der"),
}

# Landmarks de MediaPipe equivalentes, solo para mostrarlos en el panel.
INDICES_MP = {
    "front": (11, 12, 24, 23, 25, 26, 28, 32, 30, 27, 31, 29, 13, 14, 16, 15),
    "left": (25, 23, 27, 31, 29),
    "right": (26, 24, 28, 32, 30),
}
CONEXIONES = {
    "front": (
        (0, 1), (2, 3), (0, 3), (1, 2),
        (0, 12), (12, 15), (1, 13), (13, 14),
        (3, 4), (4, 9), (2, 5), (5, 6),
    ),
    "left": ((1, 0), (0, 2)),
    "right": ((1, 0), (0, 2)),
}
# En lateral, tobillo (2), punta (3) y talón (4).
CONEXIONES_PIE = {
    "front": ((6, 7), (6, 8), (7, 8), (9, 10), (9, 11), (10, 11)),
    "left": ((2, 3), (2, 4), (3, 4)),
    "right": ((2, 3), (2, 4), (3, 4)),
}
IMG_SIZE = 960
MIN_CONF = 0.5
ALTURA_VISTA = 800
ANCHO_PANEL = 420
ANCHO_VENTANA = 1500
ALTO_VENTANA = 850


def ruta_video(vista, nombre):
    carpeta = PROJECT_ROOT / "dataset" / "sq" / vista
    if nombre is None:
        videos = sorted(carpeta.glob("*.mp4"))
        if not videos:
            raise FileNotFoundError(f"No hay videos en {carpeta}")
        return videos[0]

    ruta = Path(nombre)
    if not ruta.is_file():
        ruta = carpeta / (nombre if ruta.suffix else f"{nombre}.mp4")
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el video: {ruta}")
    return ruta


def ruta_modelo(nombre):
    ruta = Path(nombre)
    if not ruta.is_file():
        ruta = MODEL_ROOT / nombre
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el modelo: {ruta}")
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


def dibujar(frame, puntos, vista, numero, nombre_modelo, viewport=None):
    alto_original, ancho_original = frame.shape[:2]
    if viewport is None:
        ancho = int(round(ancho_original * ALTURA_VISTA / alto_original))
        alto = ALTURA_VISTA
        origen_x = origen_y = 0
        vista_video = cv2.resize(frame, (ancho, alto))
    else:
        ancho, alto = viewport.width, viewport.height
        origen_x, origen_y = viewport.left, viewport.top
        vista_video = np.zeros((ALTURA_VISTA, ANCHO_VENTANA - ANCHO_PANEL, 3), dtype=np.uint8)
        reducido = cv2.resize(frame, (ancho, alto))
        vista_video[origen_y:origen_y+alto, origen_x:origen_x+ancho] = reducido
    panel = np.full((ALTURA_VISTA, ANCHO_PANEL, 3), 28, dtype=np.uint8)

    posiciones = []
    for x, y, confianza in puntos:
        if (not np.isfinite([x, y, confianza]).all()
                or (x == 0 and y == 0) or not (0 <= x <= 1 and 0 <= y <= 1)):
            posiciones.append(None)
        else:
            posiciones.append((origen_x + int(x * (ancho - 1)),
                               origen_y + int(y * (alto - 1))))
    validos = np.isfinite(puntos[:, 2]) & (puntos[:, 2] >= MIN_CONF)

    for a, b in CONEXIONES[vista]:
        if posiciones[a] is not None and posiciones[b] is not None and validos[a] and validos[b]:
            cv2.line(vista_video, posiciones[a], posiciones[b], (0, 180, 0), 2)

    # Dibujar el pie también cuando su confianza sea baja, si hay coordenadas.
    for a, b in CONEXIONES_PIE[vista]:
        if posiciones[a] is not None and posiciones[b] is not None:
            cv2.line(vista_video, posiciones[a], posiciones[b], (0, 180, 0), 2)

    cv2.putText(panel, f"SQ {vista} | YOLO directo", (12, 32),
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


def cargar_puntos(vista, video):
    ruta = PROJECT_ROOT / "keypoints" / "sq" / "yolo" / vista / f"{video.stem}.npy"
    datos = np.load(ruta, allow_pickle=False)
    if datos.ndim != 3 or datos.shape[1:] != (len(NOMBRES[vista]), 3):
        raise ValueError(f"{ruta}: shape de pose inesperada {datos.shape}")
    return datos


def abrir_video(vista, video, usar_keypoints):
    captura = cv2.VideoCapture(str(video))
    if not captura.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video}")
    try:
        fps = captura.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        datos = cargar_puntos(vista, video) if usar_keypoints else None
        if datos is not None and len(datos) != int(captura.get(cv2.CAP_PROP_FRAME_COUNT)):
            raise ValueError(f"Video/pose con frames distintos: {video}, {len(datos)}")
        return captura, fps, datos
    except Exception:
        captura.release()
        raise


def visualizar(vista, video, modelo_path, videos=None, tracking=False,
               usar_keypoints=False, export_dir=None):
    videos = list(videos) if videos else [video]
    indice_video = videos.index(video)
    modelo = None if usar_keypoints else YOLO(str(modelo_path))
    if modelo is not None:
        forma = modelo.model.yaml.get("kpt_shape", [])
        if list(forma or []) != [len(NOMBRES[vista]), 3]:
            raise ValueError(f"{modelo_path.name} tiene kpt_shape={forma}; "
                             f"la vista {vista} requiere {[len(NOMBRES[vista]), 3]}")

    captura, fps, datos = abrir_video(vista, video, usar_keypoints)
    seguidor = BarTracker(fps)
    seguidor.set_enabled(tracking)
    ventana = f"SQ YOLO - {vista} | barra exploratoria"
    estado = {"frame": None, "puntos": None, "numero": -1, "viewport": None,
              "pausado": False, "seleccionar": False, "inicio": None,
              "fin": None, "redibujar": True, "suavizar": False, "avanzar": False}

    def on_mouse(evento, x, y, flags, param):
        if not seguidor.enabled or not estado["pausado"] or not estado["seleccionar"]:
            return
        viewport = estado["viewport"]
        if viewport is None or estado["frame"] is None:
            return
        if evento == cv2.EVENT_LBUTTONDOWN and viewport.to_video(x, y) is not None:
            estado["inicio"] = (x, y)
            estado["fin"] = (x, y)
            estado["redibujar"] = True
        elif evento == cv2.EVENT_MOUSEMOVE and estado["inicio"] is not None:
            estado["fin"] = (x, y)
            estado["redibujar"] = True
        elif evento == cv2.EVENT_LBUTTONUP and estado["inicio"] is not None:
            roi = viewport.roi_from_display(estado["inicio"], (x, y))
            estado["inicio"] = estado["fin"] = None
            if roi is not None:
                seguidor.initialize(estado["frame"], estado["numero"], roi)
                estado["seleccionar"] = False
            estado["redibujar"] = True

    def exportar():
        if not seguidor.records:
            print("No hay observaciones de barra para exportar")
            return
        carpeta = Path(export_dir) if export_dir else PROJECT_ROOT / "display" / "sq" / "bar_diagnostics"
        marca = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        ruta = carpeta / f"{video.stem}_{marca}_bar.csv"
        seguidor.csv_export(ruta)
        print(f"CSV diagnóstico: {ruta}")

    try:
        delay = max(1, int(round(1000 / fps)))
        # El canvas tiene tamaño fijo. OpenCV entrega coordenadas del mouse
        # respecto a esta imagen; las bandas se manejan con VideoViewport.
        cv2.namedWindow(ventana, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(ventana, on_mouse)
        while True:
            if estado["frame"] is None or not estado["pausado"] or estado["avanzar"]:
                ok, frame = captura.read()
                if ok:
                    numero = estado["numero"] + 1
                    if datos is not None:
                        puntos = datos[numero]
                    else:
                        resultado = modelo(frame, imgsz=IMG_SIZE, verbose=False)[0]
                        puntos = puntos_del_modelo(resultado, len(NOMBRES[vista]))
                    if seguidor.enabled:
                        seguidor.update(frame, numero)
                    estado.update(frame=frame, puntos=puntos, numero=numero,
                                  viewport=VideoViewport(frame.shape[1], frame.shape[0],
                                      0, 0, ANCHO_VENTANA-ANCHO_PANEL, ALTURA_VISTA),
                                  redibujar=True)
                else:
                    estado["pausado"] = True  # permite retroceder/cambiar vídeo al terminar
                estado["avanzar"] = False

            if estado["redibujar"] and estado["frame"] is not None:
                numero = estado["numero"]
                frame = estado["frame"]
                viewport = estado["viewport"]
                imagen = dibujar(frame, estado["puntos"], vista, numero,
                                 modelo_path.name if modelo else "keypoints YOLO .npy", viewport)
                seguidor.draw(imagen, viewport, numero)
                x_panel = ANCHO_VENTANA - ANCHO_PANEL + 12
                color = (180,180,180) if not seguidor.enabled else (
                    (50,50,255) if seguidor.state == "seguimiento perdido" else (210,60,255))
                texto = seguidor.state if seguidor.enabled else "desactivado"
                y_estado = 308 if vista != "front" else 644
                cv2.putText(imagen, f"Barra: {texto}", (x_panel, y_estado),
                            cv2.FONT_HERSHEY_SIMPLEX, .58, color, 2)
                centro = seguidor.current_center(numero)
                if centro is not None:
                    mensaje = f"PUNTO BARRA: X={centro[0]:.0f} Y={centro[1]:.0f} px"
                    color_mensaje = (0,255,255)
                elif seguidor.state == "seguimiento perdido" and seguidor.enabled:
                    mensaje = "SIN PUNTO: R + S para reseleccionar"
                    color_mensaje = (50,50,255)
                elif seguidor.enabled:
                    mensaje = "SIN PUNTO: pulsa S y arrastra ROI"
                    color_mensaje = (0,255,255)
                else:
                    mensaje = "Pulsa T para marcar la barra"
                    color_mensaje = (180,180,180)
                cv2.putText(imagen, mensaje, (x_panel, y_estado+24),
                            cv2.FONT_HERSHEY_SIMPLEX, .43, color_mensaje, 2)
                cv2.putText(imagen, f"{video.stem} | {numero/fps:.2f}s | {'PAUSA' if estado['pausado'] else 'PLAY'}",
                            (x_panel, y_estado+47), cv2.FONT_HERSHEY_SIMPLEX, .48, (220,220,220), 1)
                cv2.putText(imagen, "T on/off S ROI R reset B/N paso", (x_panel,y_estado+68),
                            cv2.FONT_HERSHEY_SIMPLEX,.44,(190,190,190),1)
                if vista != "front":
                    draw_vertical_plot(imagen, seguidor, numero,
                        max(2, int(captura.get(cv2.CAP_PROP_FRAME_COUNT))), frame.shape[0],
                        (x_panel, 394, ANCHO_PANEL-26, 376), estado["suavizar"])
                if estado["inicio"] is not None and estado["fin"] is not None:
                    cv2.rectangle(imagen, estado["inicio"], estado["fin"], (210,60,255), 2)
                    medio = ((estado["inicio"][0] + estado["fin"][0])//2,
                             (estado["inicio"][1] + estado["fin"][1])//2)
                    cv2.circle(imagen, medio, 12, (0,255,255), 2, cv2.LINE_AA)
                    cv2.drawMarker(imagen, medio, (255,255,255),
                                   cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
                cv2.imshow(ventana, imagen)
                estado["redibujar"] = False

            tecla = cv2.waitKey(30 if estado["pausado"] else delay) & 0xFF
            if tecla in (27, ord("q")):
                break
            if tecla == ord(" "):
                estado["pausado"] = not estado["pausado"]
                estado["redibujar"] = True
            elif tecla == ord("t"):
                seguidor.set_enabled(not seguidor.enabled)
                estado["seleccionar"] = estado["inicio"] = estado["fin"] = None
                estado["redibujar"] = True
            elif tecla == ord("s") and seguidor.enabled and estado["frame"] is not None:
                estado["pausado"] = True
                estado["seleccionar"] = True
                estado["redibujar"] = True
            elif tecla == ord("r"):
                seguidor.reset()
                estado["seleccionar"] = estado["inicio"] = estado["fin"] = None
                estado["redibujar"] = True
            elif tecla == ord("n") and estado["pausado"]:
                estado["avanzar"] = True
            elif tecla in (ord("b"), ord(","), ord(".")) and estado["frame"] is not None:
                salto = {ord("b"): -1, ord(","): -30, ord("."): 30}[tecla]
                limite = max(1, int(captura.get(cv2.CAP_PROP_FRAME_COUNT)))
                destino = max(0, min(limite-1, estado["numero"]+salto))
                if destino != estado["numero"]:
                    captura.set(cv2.CAP_PROP_POS_FRAMES, destino)
                    seguidor.reset()
                    estado.update(frame=None, puntos=None, numero=destino-1,
                                  pausado=True, avanzar=True, seleccionar=False,
                                  inicio=None, fin=None)
            elif tecla in (ord("["), ord("]")) and len(videos) > 1:
                siguiente = indice_video + (-1 if tecla == ord("[") else 1)
                if 0 <= siguiente < len(videos):
                    if export_dir and seguidor.records:
                        exportar()
                    captura.release()
                    indice_video = siguiente
                    video = videos[indice_video]
                    captura, fps, datos = abrir_video(vista, video, usar_keypoints)
                    seguidor.change_video(fps)
                    delay = max(1, int(round(1000 / fps)))
                    estado.update(frame=None, puntos=None, numero=-1, pausado=False,
                                  avanzar=False, seleccionar=False, inicio=None, fin=None,
                                  redibujar=True)
            elif tecla == ord("m"):
                estado["suavizar"] = not estado["suavizar"]
                estado["redibujar"] = True
            elif tecla == ord("e"):
                exportar()
    finally:
        captura.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=MODELOS, help="Vista alternativa a VISTA")
    parser.add_argument("--model", help="Ruta o nombre del .pt alternativo a MODEL_PATH")
    parser.add_argument("--video", help="Ruta del mp4 o ID alternativo a VIDEO_PATH")
    parser.add_argument("--videos", nargs="+", help="Lista de IDs/rutas; [/] cambia vídeo")
    parser.add_argument("--bar-track", action="store_true", help="Iniciar con tracking activado")
    parser.add_argument("--keypoints", action="store_true", help="Usar .npy YOLO existentes solo para visualización (sin inferencia)")
    parser.add_argument("--export-dir", help="Destino opcional de CSV diagnósticos")
    args = parser.parse_args()
    if args.video and args.videos:
        parser.error("Elige --video o --videos, no ambos")
    vista = args.view or VISTA
    modelo = (args.model if args.model else
              MODEL_PATH if args.view is None else MODEL_ROOT / MODELOS[vista])
    lista = ([ruta_video(vista, nombre) for nombre in args.videos] if args.videos else
             [ruta_video(vista, str(VIDEO_PATH)) if args.view is None and args.video is None
              else ruta_video(vista, args.video)])
    visualizar(vista, lista[0], ruta_modelo(modelo), lista,
               args.bar_track, args.keypoints, args.export_dir)


if __name__ == "__main__":
    main()
