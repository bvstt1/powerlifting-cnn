"""Inspector frontal BP: banco estable y relación 2D barra-silueta central.

La máscara de persona NO identifica el pecho: su borde superior puede ser la
cabeza. Una cercanía en la imagen no demuestra contacto físico barra-pecho.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from yolo_seg_bp_side import (
    analizar_banco_video,
    dibujar_banco_fijo,
    inclinacion_barra,
    mascara_principal,
    seleccionar_banco_frame,
)


RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_SEGMENTACION = RAIZ_PROYECTO / "models" / "bp" / "bp_front_seg_v2.pt"
RUTA_POSE = RAIZ_PROYECTO / "models" / "bp" / "bp_front_skeleton_v8.pt"
VIDEO_PREDETERMINADO = RAIZ_PROYECTO / "dataset" / "bp" / "front" / "bp_591.mp4"

HOMBROS = (1, 2)  # Índices del modelo frontal de seis keypoints.
CONFIANZA_MINIMA = 0.5
COLOR_BARRA = (0, 255, 255)
COLOR_SILUETA = (255, 200, 0)
COLOR_INCIERTO = (0, 165, 255)
COLOR_CERCA = (255, 0, 255)


def perfil_central(mascara_persona, x_min, x_max):
    """Borde superior de la silueta en una franja central del atleta."""
    x_min = max(0, int(x_min))
    x_max = min(mascara_persona.shape[1] - 1, int(x_max))
    if x_max - x_min < 5:
        return None
    perfil = []
    for x in range(x_min, x_max + 1):
        ys = np.flatnonzero(mascara_persona[:, x])
        if ys.size:
            perfil.append((x, int(ys[0])))
    if len(perfil) < 0.7 * (x_max - x_min + 1):
        return None
    return int(np.median([x for x, _ in perfil])), int(np.median([y for _, y in perfil])), perfil


def zona_central(resultado_pose, mascara_persona, caja_banco):
    """Delimita la franja entre hombros; si no hay pose, usa el banco."""
    if resultado_pose.keypoints is not None and len(resultado_pose.keypoints):
        indice = int(resultado_pose.boxes.conf.argmax())
        puntos = resultado_pose.keypoints.xy[indice].cpu().numpy()
        confianza = resultado_pose.keypoints.conf
        if (confianza is not None
                and np.all(confianza[indice, list(HOMBROS)].cpu().numpy() >= CONFIANZA_MINIMA)
                and np.all(puntos[list(HOMBROS)] != 0)):
            (x1, y1), (x2, y2) = puntos[list(HOMBROS)]
            x_izq, x_der = sorted((x1, x2))
            ancho = x_der - x_izq
            if ancho >= 30:
                perfil = perfil_central(mascara_persona, x_izq + 0.25 * ancho,
                                         x_izq + 0.75 * ancho)
                if perfil is not None:
                    # Una superficie demasiado por encima de los hombros puede
                    # ser la cabeza: no llamarla pecho ni decidir contacto.
                    ambiguo = min(y1, y2) - perfil[1] > 0.35 * ancho
                    return perfil, "pose", ambiguo

    if caja_banco is None:
        return None
    x1, _, x2, _ = caja_banco
    ancho = x2 - x1
    if ancho < 0.1 * mascara_persona.shape[1]:
        return None
    centro = (x1 + x2) / 2
    perfil = perfil_central(mascara_persona, centro - 0.2 * ancho,
                             centro + 0.2 * ancho)
    return (perfil, "banco", True) if perfil is not None else None


def borde_inferior_barra(mascara_barra, perfil_persona):
    """Usa el borde observado si cubre el centro; si no, proyecta su eje."""
    observado = []
    for x, y_persona in perfil_persona:
        ys = np.flatnonzero(mascara_barra[:, x])
        if ys.size:
            observado.append((x, int(ys[-1]), y_persona))
    if len(observado) >= 0.6 * len(perfil_persona):
        separaciones = [y_persona - y_barra for _, y_barra, y_persona in observado]
        return int(np.median(separaciones)), [(x, y) for x, y, _ in observado], False

    ajuste = inclinacion_barra(mascara_barra)
    if ajuste is None:
        return None
    pendiente, ancho_visible = ajuste
    ys, xs = np.nonzero(mascara_barra)
    x_centro = int(np.median(xs))
    # Percentil de la mitad central para no tomar solo el extremo del disco.
    franja = abs(xs - x_centro) <= max(4, int(ancho_visible * 0.1))
    if not np.any(franja):
        return None
    y_borde = float(np.percentile(ys[franja], 85))
    x_objetivo = int(np.median([x for x, _ in perfil_persona]))
    distancia = min(abs(x_objetivo - xs.min()), abs(x_objetivo - xs.max()))
    if distancia > 4 * ancho_visible:
        return None

    puntos = [(x, round(y_borde + pendiente * (x - x_centro)))
              for x, _ in perfil_persona]
    separaciones = [y_persona - y_barra
                    for (_, y_persona), (_, y_barra) in zip(perfil_persona, puntos)]
    return int(np.median(separaciones)), puntos, True


def dibujar_relacion_frontal(imagen, resultado_seg, resultado_pose, caja_banco=None,
                            texto_banco=""):
    """Visualiza una relación 2D; solo una zona anatómicamente fiable se evalúa."""
    persona = mascara_principal(resultado_seg, "person", imagen.shape[:2])
    barra = mascara_principal(resultado_seg, "bar", imagen.shape[:2])
    estado = "Sin barra/persona detectada"
    if persona is not None and barra is not None:
        zona = zona_central(resultado_pose, persona, caja_banco)
        if zona is None:
            estado = "Silueta central no evaluable (pose/banco insuficientes)"
        else:
            (x_centro, y_centro, perfil), origen, ambiguo = zona
            cv2.polylines(imagen, [np.array(perfil, dtype=np.int32)], False,
                          COLOR_SILUETA, 3)
            cv2.circle(imagen, (x_centro, y_centro), 6, COLOR_SILUETA, -1)
            medida = borde_inferior_barra(barra, perfil)
            if medida is None:
                estado = "Barra central no evaluable"
            else:
                separacion, puntos_barra, proyectado = medida
                if proyectado:
                    estado = "PROYECCION 2D (baja confianza; sin juicio de contacto)"
                elif ambiguo:
                    estado = f"SILUETA CENTRAL (cabeza/torso; no es contacto): {separacion:+d}px"
                elif abs(separacion) <= max(6, round(imagen.shape[0] * 0.012)):
                    estado = f"SOLO CERCA (2D) - NO ES CONTACTO: {separacion:+d}px"
                elif separacion > 0:
                    estado = f"Separada (proyeccion 2D): {separacion}px"
                else:
                    estado = f"Solape 2D (no confirma toque): {separacion}px"
                if origen == "banco":
                    estado += " [zona estimada por banco]"

                color = COLOR_INCIERTO if proyectado or ambiguo else COLOR_BARRA
                if proyectado:
                    for i in range(0, len(puntos_barra) - 1, 24):
                        j = min(i + 12, len(puntos_barra) - 1)
                        cv2.line(imagen, puntos_barra[i], puntos_barra[j], color, 3)
                else:
                    cv2.polylines(imagen, [np.array(puntos_barra, dtype=np.int32)],
                                  False, color, 3)
                if not proyectado and not ambiguo and separacion > 0:
                    x_medio = int(np.median([x for x, _ in puntos_barra]))
                    y_medio = int(np.median([y for _, y in puntos_barra]))
                    cv2.line(imagen, (x_medio, y_medio), (x_medio, y_centro),
                             COLOR_CERCA, 2)

    alto_panel = 72 if texto_banco else 42
    cv2.rectangle(imagen, (0, 0), (imagen.shape[1], alto_panel), (0, 0, 0), -1)
    cv2.putText(imagen, estado, (12, 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if texto_banco:
        cv2.putText(imagen, texto_banco, (12, 59), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 120, 0), 2, cv2.LINE_AA)
    return estado


def main():
    parser = argparse.ArgumentParser(description="Inspector 2D BP frontal")
    parser.add_argument("video", nargs="?", type=Path, default=VIDEO_PREDETERMINADO,
                        help=f"Video frontal (por defecto: {VIDEO_PREDETERMINADO.relative_to(RAIZ_PROYECTO)})")
    args = parser.parse_args()

    modelo_seg = YOLO(str(RUTA_SEGMENTACION))
    modelo_pose = YOLO(str(RUTA_POSE))
    captura = cv2.VideoCapture(str(args.video))
    if not captura.isOpened():
        raise OSError(f"No se pudo abrir el video: {args.video}")

    try:
        referencia, banco_fijo = analizar_banco_video(captura, modelo_seg)
        if not captura.set(cv2.CAP_PROP_POS_FRAMES, 0):
            raise OSError(f"No se pudo reiniciar el video: {args.video}")
        if referencia is not None:
            print(f"Banco: mejor frame={referencia['frame']}, "
                  f"confianza={referencia['confianza']:.1%}, "
                  f"referencia fija={'si' if banco_fijo else 'no'}")

        ventana = "BP frontal: barra y silueta (Q para salir)"
        while True:
            disponible, frame = captura.read()
            if not disponible:
                break
            resultado_seg = modelo_seg(frame, verbose=False, show=False, save=False)[0]
            resultado_pose = modelo_pose(frame, imgsz=960, verbose=False,
                                         show=False, save=False)[0]
            caja_banco = seleccionar_banco_frame(resultado_seg, referencia, banco_fijo)

            imagen = frame.copy()
            if banco_fijo and referencia is not None:
                persona = mascara_principal(resultado_seg, "person", frame.shape[:2])
                dibujar_banco_fijo(imagen, referencia, persona)
            imagen = resultado_seg.plot(img=imagen)
            texto_banco = ""
            if banco_fijo and referencia is not None:
                texto_banco = (f"Banco fijo: frame {referencia['frame']}, "
                               f"confianza {referencia['confianza']:.1%}")
            dibujar_relacion_frontal(imagen, resultado_seg, resultado_pose, caja_banco,
                                    texto_banco)

            alto, ancho = imagen.shape[:2]
            escala = min(640 / ancho, 360 / alto, 1)
            if escala < 1:
                imagen = cv2.resize(imagen, (round(ancho * escala), round(alto * escala)),
                                    interpolation=cv2.INTER_AREA)
            cv2.imshow(ventana, imagen)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        captura.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
