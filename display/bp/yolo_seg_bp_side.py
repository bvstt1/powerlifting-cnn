"""Inspección visual de proximidad barra-pecho en press de banca lateral.

La barra puede estar oculta entre la mano y el pecho: la extrapolación de su
inclinación visible es una estimación 2D, no una prueba de contacto físico.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_SEGMENTACION = RAIZ_PROYECTO / "models" / "bp" / "bp_side_seg_v1.pt"
RUTA_POSE = RAIZ_PROYECTO / "models" / "bp" / "bp_side_skeleton_v2.pt"
VIDEO_PREDETERMINADO = RAIZ_PROYECTO / "dataset" / "bp" / "left" / "bp_345.mp4"

# Orden de keypoints del modelo lateral: 2=hombro, 3=codo.
HOMBRO = 2
CODO = 3
CONFIANZA_MINIMA = 0.5
COLOR_BARRA = (0, 255, 255)
COLOR_PECHO = (255, 200, 0)
COLOR_CERCA = (255, 0, 255)  # Magenta: proximidad, nunca contacto confirmado
COLOR_LEJOS = (0, 0, 255)
COLOR_PROYECCION_LEJANA = (0, 165, 255)
COLOR_BANCO = (255, 120, 0)


def mascara_principal(resultado, nombre_clase, forma):
    """Devuelve la máscara de mayor confianza de una clase, o None."""
    if resultado.boxes is None or resultado.masks is None:
        return None

    candidatos = [
        i for i, caja in enumerate(resultado.boxes)
        if resultado.names[int(caja.cls.item())] == nombre_clase
        and i < len(resultado.masks.xy) and len(resultado.masks.xy[i]) >= 3
    ]
    if not candidatos:
        return None

    indice = max(candidatos, key=lambda i: float(resultado.boxes.conf[i]))
    mascara = np.zeros(forma, dtype=np.uint8)
    cv2.fillPoly(mascara, [resultado.masks.xy[indice].astype(np.int32)], 1)
    return mascara


def zona_pecho(resultado_pose, mascara_persona):
    """Perfil superior de la silueta en la zona del tórax (entre hombro y codo)."""
    if resultado_pose.keypoints is None or len(resultado_pose.keypoints) == 0:
        return None
    indice = int(resultado_pose.boxes.conf.argmax())
    puntos = resultado_pose.keypoints.xy[indice].cpu().numpy()
    confianza = resultado_pose.keypoints.conf
    if confianza is None or np.any(
        confianza[indice, [HOMBRO, CODO]].cpu().numpy() < CONFIANZA_MINIMA
    ) or np.any(puntos[[HOMBRO, CODO]] == 0):
        return None

    # Restringir al torso central para evitar medir el brazo o la mano.
    x_hombro, x_codo = puntos[[HOMBRO, CODO], 0]
    x_inicio = x_hombro + 0.45 * (x_codo - x_hombro)
    x_fin = x_hombro + 0.75 * (x_codo - x_hombro)
    ancho = mascara_persona.shape[1]
    x_min = max(0, int(min(x_inicio, x_fin)))
    x_max = min(ancho - 1, int(max(x_inicio, x_fin)))
    return perfil_pecho(mascara_persona, x_min, x_max)


def perfil_pecho(mascara_persona, x_min, x_max):
    """Extrae la superficie superior de la persona en una franja del torso."""
    if x_max - x_min < 5:
        return None

    # Mediana del borde superior: evita un único píxel aislado de segmentación.
    perfil = []
    for x in range(x_min, x_max + 1):
        ys = np.flatnonzero(mascara_persona[:, x])
        if ys.size:
            perfil.append((x, int(ys[0])))
    if len(perfil) < (x_max - x_min + 1) * 0.7:
        return None
    x_pecho = int(np.median([p[0] for p in perfil]))
    y_pecho = int(np.median([p[1] for p in perfil]))
    return x_pecho, y_pecho, perfil


def zona_pecho_sin_pose(caja_banco, mascara_persona, mascara_barra):
    """Estima una franja del torso respecto del banco cuando falla la pose."""
    if caja_banco is None:
        return None
    x1, _, x2, _ = caja_banco
    ancho_banco = x2 - x1
    if ancho_banco < mascara_persona.shape[1] * 0.1:
        return None

    # La zona torácica suele estar cerca del centro del banco. El lado donde
    # aparece la barra determina hacia qué extremo desplazar ligeramente la
    # franja; no se confunde con la mano en el extremo de la máscara humana.
    centro = (x1 + x2) / 2
    xs_barra = np.nonzero(mascara_barra)[1]
    if not xs_barra.size:
        return None
    direccion = 1 if np.median(xs_barra) > centro else -1
    limite_a = centro - direccion * 0.10 * ancho_banco
    limite_b = centro + direccion * 0.15 * ancho_banco
    x_min = max(0, int(min(limite_a, limite_b)))
    x_max = min(mascara_persona.shape[1] - 1, int(max(limite_a, limite_b)))
    return perfil_pecho(mascara_persona, x_min, x_max)


def borde_barra_cercano(mascara_barra, x_pecho):
    """Altura del borde inferior de la barra visible junto al extremo proximal."""
    ys, xs = np.nonzero(mascara_barra)
    if not xs.size:
        return None
    x_min, x_max = int(xs.min()), int(xs.max())
    franja = max(4, int((x_max - x_min) * 0.15))
    if abs(x_pecho - x_min) < abs(x_pecho - x_max):
        cercano = xs <= x_min + franja
    else:
        cercano = xs >= x_max - franja
    if not np.any(cercano):
        return None
    x_barra = int(np.median(xs[cercano]))
    y_barra = int(np.median(ys[cercano]))
    grosor = int(np.percentile(ys[cercano], 85) - y_barra)
    return x_barra, y_barra, max(0, grosor)


def inclinacion_barra(mascara_barra):
    """Ajusta la pendiente y(x) a la línea central de la máscara visible."""
    ys, xs = np.nonzero(mascara_barra)
    if not xs.size:
        return None
    x_min, x_max = int(xs.min()), int(xs.max())
    ancho = x_max - x_min
    if ancho < 30:
        return None

    # Excluir ambos extremos, donde los discos deforman más el perfil.
    centros = []
    for x in range(x_min + int(ancho * 0.15), x_max - int(ancho * 0.15) + 1):
        columna = np.flatnonzero(mascara_barra[:, x])
        if columna.size:
            centros.append((x, float(np.median(columna))))
    if len(centros) < ancho * 0.5:
        return None

    valores = np.array(centros)
    pendiente, origen = np.polyfit(valores[:, 0], valores[:, 1], 1)
    residuo = np.median(np.abs(valores[:, 1] - (pendiente * valores[:, 0] + origen)))
    # Antes que inventar un eje horizontal, declarar no evaluable una máscara
    # demasiado deformada o con una pendiente poco fiable.
    if abs(pendiente) > 0.35 or residuo > mascara_barra.shape[0] * 0.02:
        return None
    return float(pendiente), ancho


def dibujar_proximidad(imagen, resultado_seg, resultado_pose, caja_banco=None,
                      texto_banco=""):
    """Muestra borde inclinado extrapolado, perfil del pecho y separación 2D."""
    mascara_barra = mascara_principal(resultado_seg, "bar", imagen.shape[:2])
    mascara_persona = mascara_principal(resultado_seg, "person", imagen.shape[:2])
    estado = "Sin barra/persona detectada"
    if mascara_barra is not None and mascara_persona is not None:
        pecho = zona_pecho(resultado_pose, mascara_persona)
        estimado = pecho is None
        if estimado:
            pecho = zona_pecho_sin_pose(caja_banco, mascara_persona, mascara_barra)
        if pecho is None:
            estado = "Pecho no evaluable (sin pose ni banco fiable)"
        else:
            x_pecho, y_pecho, perfil = pecho
            barra = borde_barra_cercano(mascara_barra, x_pecho)
            inclinacion = inclinacion_barra(mascara_barra)
            if barra is None or inclinacion is None:
                estado = "Inclinacion de barra no evaluable"
            else:
                x_barra, y_barra, radio = barra
                pendiente, ancho_visible = inclinacion
                distancia_proyectada = abs(x_pecho - x_barra)
                if distancia_proyectada > 4 * ancho_visible:
                    estado = "Barra demasiado lejos para proyectar"
                else:
                    lejana = distancia_proyectada > 2 * ancho_visible

                    def y_proyectada(x):
                        return y_barra + radio + pendiente * (x - x_barra)

                    separacion = round(y_pecho - y_proyectada(x_pecho))
                    tolerancia = max(6, round(imagen.shape[0] * 0.012))
                    if lejana:
                        estado = "PROYECCION LEJANA (baja confianza; sin juicio de contacto)"
                        color = COLOR_PROYECCION_LEJANA
                    elif abs(separacion) <= tolerancia:
                        estado = f"SOLO CERCA (2D) - NO ES CONTACTO: {separacion:+d}px"
                        color = COLOR_CERCA
                    elif separacion > 0:
                        estado = f"Separada (proyeccion 2D): {separacion}px"
                        color = COLOR_LEJOS
                    else:
                        estado = f"Solape proyectado (no confirma toque): {separacion}px"
                        color = COLOR_LEJOS
                    if estimado:
                        estado += " [pecho estimado por banco]"

                    cv2.polylines(imagen, [np.array(perfil, dtype=np.int32)], False,
                                  COLOR_PECHO, 3)
                    cv2.circle(imagen, (x_pecho, y_pecho), 7, COLOR_PECHO, -1)
                    cv2.circle(imagen, (x_barra, round(y_proyectada(x_barra))),
                               7, COLOR_BARRA, -1)

                    # Si la proyección entra en la silueta del torso, detenerla
                    # en el primer borde; no dibujarla atravesando el cuerpo.
                    x_final = x_pecho
                    y_tope = None
                    perfil_hacia_pecho = sorted(perfil, key=lambda p: abs(p[0] - x_barra))
                    for x, y_borde in perfil_hacia_pecho:
                        if abs(x - x_barra) > abs(x_pecho - x_barra):
                            break
                        if y_proyectada(x) >= y_borde:
                            x_final = x
                            y_tope = y_borde
                            break

                    paso = 1 if x_final >= x_barra else -1
                    color_linea = COLOR_PROYECCION_LEJANA if lejana else COLOR_BARRA
                    for x in range(x_barra, x_final, paso * 24):
                        destino = x + paso * min(12, abs(x_final - x))
                        y_destino = round(y_proyectada(destino))
                        if destino == x_final and y_tope is not None:
                            y_destino = min(y_destino, y_tope)
                        cv2.line(imagen, (x, round(y_proyectada(x))),
                                 (destino, y_destino), color_linea, 3)
                    if not lejana and x_final == x_pecho and separacion > 0:
                        cv2.line(imagen, (x_pecho, round(y_proyectada(x_pecho))),
                                 (x_pecho, y_pecho), color, 3)

    alto_panel = 72 if texto_banco else 42
    cv2.rectangle(imagen, (0, 0), (imagen.shape[1], alto_panel), (0, 0, 0), -1)
    cv2.putText(imagen, estado, (12, 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if texto_banco:
        cv2.putText(imagen, texto_banco, (12, 59), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, COLOR_BANCO, 2, cv2.LINE_AA)
    return estado


def mejor_banco(resultado_seg):
    """Escoge una sola instancia del banco: la de mayor confianza YOLO."""
    if resultado_seg.boxes is None or resultado_seg.masks is None:
        return None
    indices = [
        i for i, caja in enumerate(resultado_seg.boxes)
        if resultado_seg.names[int(caja.cls.item())] == "bench"
        and i < len(resultado_seg.masks.xy)
        and len(resultado_seg.masks.xy[i]) >= 3
    ]
    return max(indices, key=lambda i: float(resultado_seg.boxes.conf[i])) if indices else None


def solapamiento_cajas(caja_a, caja_b):
    x1 = max(caja_a[0], caja_b[0])
    y1 = max(caja_a[1], caja_b[1])
    x2 = min(caja_a[2], caja_b[2])
    y2 = min(caja_a[3], caja_b[3])
    interseccion = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, caja_a[2] - caja_a[0]) * max(0, caja_a[3] - caja_a[1])
    area_b = max(0, caja_b[2] - caja_b[0]) * max(0, caja_b[3] - caja_b[1])
    union = area_a + area_b - interseccion
    return interseccion / union if union else 0


def analizar_banco_video(captura, modelo_seg):
    """Busca el frame de mayor confianza sin mezclarlo con otros videos."""
    cajas = []
    referencia = None
    frame_indice = 0
    while True:
        disponible, frame = captura.read()
        if not disponible:
            break
        resultado = modelo_seg(frame, verbose=False, show=False, save=False)[0]
        indice = mejor_banco(resultado)
        if indice is not None:
            caja = resultado.boxes.xyxy[indice].cpu().numpy().copy()
            confianza = float(resultado.boxes.conf[indice])
            cajas.append(caja)
            if referencia is None or confianza > referencia["confianza"]:
                referencia = {
                    "frame": frame_indice,
                    "confianza": confianza,
                    "caja": caja,
                    "poligono": resultado.masks.xy[indice].astype(np.int32).copy(),
                }
        frame_indice += 1

    if referencia is None:
        return None, False
    mediana = np.median(np.array(cajas), axis=0)
    fluctuacion = np.mean([solapamiento_cajas(caja, mediana) < 0.8 for caja in cajas])
    fijo = len(cajas) >= 5 and fluctuacion >= 0.20
    return referencia, fijo


def seleccionar_banco_frame(resultado_seg, referencia, fijo):
    """Elimina duplicados y devuelve la caja usada para ubicar el pecho."""
    indice = mejor_banco(resultado_seg)
    if resultado_seg.boxes is not None:
        conservar = [
            i for i, caja in enumerate(resultado_seg.boxes)
            if resultado_seg.names[int(caja.cls.item())] != "bench"
        ]
        if not fijo and indice is not None:
            conservar.append(indice)
        resultado_seg.boxes = resultado_seg.boxes[conservar]
        if resultado_seg.masks is not None:
            resultado_seg.masks = resultado_seg.masks[conservar]

    if fijo or indice is None:
        return referencia["caja"] if referencia is not None else None
    return resultado_seg.boxes.xyxy[-1].cpu().numpy()


def dibujar_banco_fijo(imagen, referencia, mascara_persona):
    """Dibuja el banco por detrás de la silueta de la persona."""
    poligono = referencia["poligono"]
    zona_banco = np.zeros(imagen.shape[:2], dtype=np.uint8)
    cv2.fillPoly(zona_banco, [poligono], 1)
    contorno = np.zeros_like(zona_banco)
    cv2.polylines(contorno, [poligono], True, 1, 3)

    if mascara_persona is not None:
        # Un pequeño margen evita que el contorno del banco atraviese el cuerpo.
        persona = cv2.dilate(mascara_persona, np.ones((5, 5), dtype=np.uint8))
        zona_banco[persona > 0] = 0
        contorno[persona > 0] = 0

    pixeles_banco = zona_banco > 0
    color = np.full_like(imagen, COLOR_BANCO)
    imagen[pixeles_banco] = cv2.addWeighted(
        imagen[pixeles_banco], 0.8, color[pixeles_banco], 0.2, 0
    )
    imagen[contorno > 0] = COLOR_BANCO


def main():
    parser = argparse.ArgumentParser(description="Inspeccionar proximidad barra-pecho BP lateral")
    parser.add_argument(
        "video", nargs="?", type=Path, default=VIDEO_PREDETERMINADO,
        help=f"Video lateral (por defecto: {VIDEO_PREDETERMINADO.relative_to(RAIZ_PROYECTO)})",
    )
    args = parser.parse_args()

    modelo_seg = YOLO(str(RUTA_SEGMENTACION))
    modelo_pose = YOLO(str(RUTA_POSE))
    captura = cv2.VideoCapture(str(args.video))
    if not captura.isOpened():
        raise OSError(f"No se pudo abrir el video: {args.video}")

    ventana = "BP lateral: barra y pecho (Q para salir)"
    try:
        referencia, banco_fijo = analizar_banco_video(captura, modelo_seg)
        if not captura.set(cv2.CAP_PROP_POS_FRAMES, 0):
            raise OSError(f"No se pudo reiniciar el video: {args.video}")
        if referencia is not None:
            print(f"Banco: mejor frame={referencia['frame']}, "
                  f"confianza={referencia['confianza']:.1%}, "
                  f"referencia fija={'si' if banco_fijo else 'no'}")
        while True:
            disponible, frame = captura.read()
            if not disponible:
                break

            resultado_seg = modelo_seg(frame, verbose=False, show=False, save=False)[0]
            resultado_pose = modelo_pose(frame, verbose=False, show=False, save=False)[0]
            caja_banco = seleccionar_banco_frame(resultado_seg, referencia, banco_fijo)
            imagen = frame.copy()
            if banco_fijo and referencia is not None:
                mascara_persona = mascara_principal(resultado_seg, "person", frame.shape[:2])
                dibujar_banco_fijo(imagen, referencia, mascara_persona)
            imagen = resultado_seg.plot(img=imagen)
            texto_banco = ""
            if banco_fijo and referencia is not None:
                texto_banco = (f"Banco fijo: frame {referencia['frame']}, "
                               f"confianza {referencia['confianza']:.1%}")
            dibujar_proximidad(imagen, resultado_seg, resultado_pose, caja_banco,
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
