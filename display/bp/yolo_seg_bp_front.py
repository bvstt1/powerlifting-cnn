"""Inspector frontal BP: banco estable y relación 2D barra-silueta central.

La máscara de persona NO identifica el pecho: su borde superior puede ser la
cabeza. Una cercanía en la imagen no demuestra contacto físico barra-pecho.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_SEGMENTACION = RAIZ_PROYECTO / "models" / "bp" / "bp_front_seg_v2.pt"
RUTA_POSE = RAIZ_PROYECTO / "models" / "bp" / "bp_front_skeleton_v8.pt"
VIDEO_PREDETERMINADO = RAIZ_PROYECTO / "dataset" / "bp" / "front" / "bp_518.mp4"

HOMBROS = (1, 2)  # Índices del modelo frontal de seis keypoints.
MUNECAS = (5, 4)  # Muñeca izquierda y derecha en el mismo modelo.
CONFIANZA_MINIMA = 0.5
CONFIANZA_ANCLA_BARRA = 0.65
MAX_FRAMES_ESTIMADOS = 6
COLOR_BARRA = (0, 255, 255)
COLOR_SILUETA = (255, 200, 0)
COLOR_INCIERTO = (0, 165, 255)
COLOR_ESTIMADA = (0, 120, 255)
COLOR_CERCA = (255, 0, 255)
COLOR_BANCO = (255, 120, 0)


def mascara_principal(resultado, nombre_clase, forma):
    """Convierte en binaria la máscara más confiable de una clase."""
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


def puntos_munecas(resultado_pose):
    """Devuelve las dos muñecas ordenadas de izquierda a derecha."""
    if resultado_pose.keypoints is None or len(resultado_pose.keypoints) == 0:
        return None
    indice = int(resultado_pose.boxes.conf.argmax())
    puntos = resultado_pose.keypoints.xy[indice].cpu().numpy()
    confianza = resultado_pose.keypoints.conf
    if (confianza is None
            or np.any(confianza[indice, list(MUNECAS)].cpu().numpy() < CONFIANZA_MINIMA)
            or np.any(puntos[list(MUNECAS)] == 0)):
        return None
    munecas = puntos[list(MUNECAS)]
    munecas = munecas[np.argsort(munecas[:, 0])]
    return munecas if munecas[1, 0] - munecas[0, 0] >= 30 else None


def intervalo_munecas(resultado_pose):
    """Devuelve la zona horizontal entre ambas muñecas si son fiables."""
    puntos = puntos_munecas(resultado_pose)
    return (puntos[0, 0], puntos[1, 0]) if puntos is not None else None


def seleccion_barras(resultado_seg, resultado_pose):
    """Conserva barras mayormente dentro de persona o entre muñecas.

    Si dos máscaras cubren casi el mismo rectángulo, representan una única
    detección del mismo objeto y se mantiene la de mayor confianza.
    """
    if resultado_seg.boxes is None or resultado_seg.masks is None:
        return 0
    personas = [
        i for i, caja in enumerate(resultado_seg.boxes)
        if resultado_seg.names[int(caja.cls.item())] == "person"
    ]
    caja_persona = None
    if personas:
        indice = max(personas, key=lambda i: float(resultado_seg.boxes.conf[i]))
        caja_persona = resultado_seg.boxes.xyxy[indice].cpu().numpy()
    munecas = intervalo_munecas(resultado_pose)
    puntos_manos = puntos_munecas(resultado_pose)
    altura_manos = float(np.mean(puntos_manos[:, 1])) if puntos_manos is not None else None

    candidatos = []
    mascaras = {}
    forma = resultado_seg.orig_shape
    for i, caja in enumerate(resultado_seg.boxes):
        if resultado_seg.names[int(caja.cls.item())] != "bar":
            continue
        if i >= len(resultado_seg.masks.xy) or len(resultado_seg.masks.xy[i]) < 3:
            continue
        mascara = np.zeros(forma, dtype=np.uint8)
        cv2.fillPoly(mascara, [resultado_seg.masks.xy[i].astype(np.int32)], 1)
        ys, xs = np.nonzero(mascara)
        if not xs.size:
            continue
        # Una estructura inferior situada bastante por DEBAJO de ambas manos
        # no es la barra que estas sujetan, aunque YOLO la llame BAR y su caja
        # caiga sobre la caja amplia de la persona (bp_662/663 RAW a 960).
        if (altura_manos is not None
                and (float(np.median(ys)) - altura_manos) > 0.03 * forma[0]):
            continue
        dentro_persona = (caja_persona is not None and np.mean(
            (xs >= caja_persona[0]) & (xs <= caja_persona[2])
            & (ys >= caja_persona[1]) & (ys <= caja_persona[3])
        ) >= 0.5)
        entre_munecas = (munecas is not None and np.mean(
            (xs >= munecas[0]) & (xs <= munecas[1])
        ) >= 0.5)
        if dentro_persona or entre_munecas:
            candidatos.append(i)
            mascaras[i] = mascara

    candidatos.sort(key=lambda i: float(resultado_seg.boxes.conf[i]), reverse=True)
    elegidas = []
    for indice in candidatos:
        if all(solapamiento_mascaras(mascaras[indice], mascaras[prev]) < 0.85
               for prev in elegidas):
            elegidas.append(indice)

    conservar = [
        i for i, caja in enumerate(resultado_seg.boxes)
        if resultado_seg.names[int(caja.cls.item())] != "bar"
    ] + elegidas
    resultado_seg.boxes = resultado_seg.boxes[conservar]
    resultado_seg.masks = resultado_seg.masks[conservar]
    return len(elegidas)


def solapamiento_mascaras(mascara_a, mascara_b):
    """Intersección / máscara menor: duplicados, no solo cajas próximas."""
    minimo = min(np.count_nonzero(mascara_a), np.count_nonzero(mascara_b))
    return (np.count_nonzero(mascara_a & mascara_b) / minimo) if minimo else 0


def mascara_barras(resultado_seg, forma):
    """Une los fragmentos seleccionados de barra para medirlos juntos."""
    if resultado_seg.boxes is None or resultado_seg.masks is None:
        return None
    mascara = np.zeros(forma, dtype=np.uint8)
    hay_barras = False
    for i, caja in enumerate(resultado_seg.boxes):
        if resultado_seg.names[int(caja.cls.item())] == "bar":
            cv2.fillPoly(mascara, [resultado_seg.masks.xy[i].astype(np.int32)], 1)
            hay_barras = True
    return mascara if hay_barras else None


def inclinacion_barra(mascara_barra):
    """Ajusta el eje visible sin inventar una pendiente si la máscara falla."""
    ys, xs = np.nonzero(mascara_barra)
    if not xs.size:
        return None
    x_min, x_max = int(xs.min()), int(xs.max())
    ancho = x_max - x_min
    if ancho < 30:
        return None
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
    if abs(pendiente) > 0.35 or residuo > mascara_barra.shape[0] * 0.02:
        return None
    return float(pendiente), ancho


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
                            texto_banco="", numero_barras=0, estimacion_barra=None,
                            barras_raw=0):
    """Visualiza una relación 2D; solo una zona anatómicamente fiable se evalúa."""
    persona = mascara_principal(resultado_seg, "person", imagen.shape[:2])
    barra = mascara_barras(resultado_seg, imagen.shape[:2])
    if persona is None:
        estado = ("Sin persona segmentada; BAR detectada por visor"
                  if barras_raw else "Sin persona detectada")
    elif barra is None:
        estado = ("BAR detectada por visor; identidad no verificada"
                  if barras_raw else "BAR no detectada por visor")
    else:
        estado = "Relación barra-silueta no evaluable"
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
                if numero_barras > 1:
                    estado = f"{numero_barras} barras internas distintas (medida ambigua)"
                elif proyectado:
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

                color = COLOR_INCIERTO if proyectado or ambiguo or numero_barras > 1 else COLOR_BARRA
                if proyectado:
                    for i in range(0, len(puntos_barra) - 1, 24):
                        j = min(i + 12, len(puntos_barra) - 1)
                        cv2.line(imagen, puntos_barra[i], puntos_barra[j], color, 3)
                else:
                    cv2.polylines(imagen, [np.array(puntos_barra, dtype=np.int32)],
                                  False, color, 3)
                if not proyectado and not ambiguo and numero_barras == 1 and separacion > 0:
                    x_medio = int(np.median([x for x, _ in puntos_barra]))
                    y_medio = int(np.median([y for _, y in puntos_barra]))
                    cv2.line(imagen, (x_medio, y_medio), (x_medio, y_centro),
                             COLOR_CERCA, 2)

    if barra is None and estimacion_barra is not None:
        puntos = puntos_barra_estimada(resultado_pose, estimacion_barra, imagen.shape[:2])
        if puntos is not None:
            (x1, y1), (x2, y2) = puntos
            for x in range(x1, x2, 24):
                destino = min(x + 12, x2)
                y_inicio = round(y1 + (y2 - y1) * (x - x1) / (x2 - x1))
                y_destino = round(y1 + (y2 - y1) * (destino - x1) / (x2 - x1))
                cv2.line(imagen, (x, y_inicio), (destino, y_destino),
                         COLOR_ESTIMADA, 3)
            anterior, posterior = estimacion_barra["anclas"]
            estado = (f"BARRA ESTIMADA POR MUNECAS ({anterior}-{posterior}): "
                      "sin deteccion; contacto NO evaluable")

    alto_panel = 72 if texto_banco else 42
    cv2.rectangle(imagen, (0, 0), (imagen.shape[1], alto_panel), (0, 0, 0), -1)
    cv2.putText(imagen, estado, (12, 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if texto_banco:
        cv2.putText(imagen, texto_banco, (12, 59), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 120, 0), 2, cv2.LINE_AA)
    return estado


def mejor_banco(resultado_seg):
    """Elige una única máscara válida del banco, la de mayor confianza."""
    if resultado_seg.boxes is None or resultado_seg.masks is None:
        return None
    candidatos = [
        i for i, caja in enumerate(resultado_seg.boxes)
        if resultado_seg.names[int(caja.cls.item())] == "bench"
        and i < len(resultado_seg.masks.xy)
        and len(resultado_seg.masks.xy[i]) >= 3
    ]
    return max(candidatos, key=lambda i: float(resultado_seg.boxes.conf[i])) if candidatos else None


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


def ancla_barra(resultado_seg):
    """Barra visible, centrada en la persona y sin recorte de imagen."""
    if resultado_seg.boxes is None or resultado_seg.masks is None:
        return None
    personas = [i for i, caja in enumerate(resultado_seg.boxes)
                if resultado_seg.names[int(caja.cls.item())] == "person"]
    if not personas:
        return None
    indice_persona = max(personas, key=lambda i: float(resultado_seg.boxes.conf[i]))
    persona = resultado_seg.boxes.xyxy[indice_persona].cpu().numpy()
    alto = resultado_seg.orig_shape[0]
    margen = max(8, round(0.01 * alto))
    candidatas = []
    for i, caja in enumerate(resultado_seg.boxes):
        if (resultado_seg.names[int(caja.cls.item())] != "bar"
                or float(caja.conf.item()) < CONFIANZA_ANCLA_BARRA
                or i >= len(resultado_seg.masks.xy)
                or len(resultado_seg.masks.xy[i]) < 3):
            continue
        barra = caja.xyxy[0].cpu().numpy()
        if barra[1] < margen or barra[3] > alto - margen:
            continue
        x1, y1 = max(persona[0], barra[0]), max(persona[1], barra[1])
        x2, y2 = min(persona[2], barra[2]), min(persona[3], barra[3])
        area_barra = max(0, barra[2] - barra[0]) * max(0, barra[3] - barra[1])
        interseccion = max(0, x2 - x1) * max(0, y2 - y1)
        if area_barra and interseccion / area_barra >= 0.5:
            candidatas.append((float(caja.conf.item()), barra))
    if not candidatas:
        return None
    _, barra = max(candidatas, key=lambda item: item[0])
    return {
        "x": float((barra[0] + barra[2]) / 2),
        "y": float((barra[1] + barra[3]) / 2),
        "ancho": float(barra[2] - barra[0]),
    }


def analizar_banco_video(captura, modelo_seg, anclas=None, barra_detectada=None):
    """Busca banco y, opcionalmente, anclas de barra del video abierto."""
    cajas = []
    referencia = None
    frame_indice = 0
    while True:
        disponible, frame = captura.read()
        if not disponible:
            break
        resultado = modelo_seg(frame, verbose=False, show=False, save=False)[0]
        if anclas is not None:
            anclas.append(ancla_barra(resultado))
        if barra_detectada is not None:
            barra_detectada.append(any(
                resultado.names[int(caja.cls.item())] == "bar"
                for caja in resultado.boxes
            ))
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
    caja_mediana = np.median(np.array(cajas), axis=0)
    fluctuacion = np.mean([solapamiento_cajas(caja, caja_mediana) < 0.8
                           for caja in cajas])
    fijo = len(cajas) >= 5 and fluctuacion >= 0.20
    return referencia, fijo


def preparar_estimaciones(captura, modelo_pose, anclas, barra_detectada):
    """Interpola solo huecos cortos entre dos barras visibles y fiables."""
    estimaciones = {}
    indice = 0
    while indice < len(anclas):
        if anclas[indice] is not None:
            indice += 1
            continue
        comienzo = indice
        while indice < len(anclas) and anclas[indice] is None:
            indice += 1
        anterior, posterior = comienzo - 1, indice
        if (anterior < 0 or posterior >= len(anclas)
                or posterior - anterior - 1 > MAX_FRAMES_ESTIMADOS
                or any(barra_detectada[comienzo:posterior])):
            continue

        extremos = []
        for frame_indice in (anterior, posterior):
            if not captura.set(cv2.CAP_PROP_POS_FRAMES, frame_indice):
                extremos = []
                break
            disponible, frame = captura.read()
            if not disponible:
                extremos = []
                break
            pose = modelo_pose(frame, imgsz=960, verbose=False, show=False,
                               save=False)[0]
            munecas = puntos_munecas(pose)
            if munecas is None:
                extremos = []
                break
            ancla = anclas[frame_indice]
            ancho_munecas = float(munecas[1, 0] - munecas[0, 0])
            if (not munecas[0, 0] <= ancla["x"] <= munecas[1, 0]
                    or ancla["ancho"] < 0.2 * ancho_munecas):
                extremos = []
                break
            extremos.append((float(ancla["y"] - np.mean(munecas[:, 1])),
                             ancho_munecas, frame.shape[0]))

        if len(extremos) != 2:
            continue
        (offset_a, ancho_a, alto), (offset_b, ancho_b, _) = extremos
        if (abs(offset_a - offset_b) > max(12, 0.03 * alto)
                or abs(ancho_a - ancho_b) > 0.35 * max(ancho_a, ancho_b)):
            continue

        for frame_indice in range(comienzo, posterior):
            fraccion = (frame_indice - anterior) / (posterior - anterior)
            estimaciones[frame_indice] = {
                "offset": (1 - fraccion) * offset_a + fraccion * offset_b,
                "ancho_munecas": (1 - fraccion) * ancho_a + fraccion * ancho_b,
                "anclas": (anterior, posterior),
            }
    return estimaciones


def puntos_barra_estimada(resultado_pose, estimacion, forma):
    """Traza entre muñecas, desplazado al nivel de barra de ambos extremos."""
    munecas = puntos_munecas(resultado_pose)
    if munecas is None:
        return None
    ancho = float(munecas[1, 0] - munecas[0, 0])
    if abs(ancho - estimacion["ancho_munecas"]) > 0.35 * max(ancho, estimacion["ancho_munecas"]):
        return None
    puntos = [(round(float(x)), round(float(y + estimacion["offset"])))
              for x, y in munecas]
    alto, ancho_frame = forma
    if any(x < 0 or x >= ancho_frame or y < 8 or y >= alto - 8 for x, y in puntos):
        return None
    return puntos


def seleccionar_banco_frame(resultado_seg, referencia, fijo):
    """Deja como máximo un banco en el resultado dibujable."""
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
    """Muestra la máscara congelada sin pintar por encima de la persona."""
    poligono = referencia["poligono"]
    zona_banco = np.zeros(imagen.shape[:2], dtype=np.uint8)
    cv2.fillPoly(zona_banco, [poligono], 1)
    contorno = np.zeros_like(zona_banco)
    cv2.polylines(contorno, [poligono], True, 1, 3)
    if mascara_persona is not None:
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
        anclas = []
        barra_detectada = []
        referencia, banco_fijo = analizar_banco_video(
            captura, modelo_seg, anclas, barra_detectada
        )
        estimaciones = preparar_estimaciones(
            captura, modelo_pose, anclas, barra_detectada
        )
        if not captura.set(cv2.CAP_PROP_POS_FRAMES, 0):
            raise OSError(f"No se pudo reiniciar el video: {args.video}")
        if referencia is not None:
            print(f"Banco: mejor frame={referencia['frame']}, "
                  f"confianza={referencia['confianza']:.1%}, "
                  f"referencia fija={'si' if banco_fijo else 'no'}")
        if estimaciones:
            print(f"Barra estimada por muñecas en {len(estimaciones)} frames; "
                  "sin reemplazar detecciones originales")

        ventana = "BP frontal: barra y silueta (Q para salir)"
        frame_indice = 0
        while True:
            disponible, frame = captura.read()
            if not disponible:
                break
            resultado_seg = modelo_seg(frame, verbose=False, show=False, save=False)[0]
            resultado_pose = modelo_pose(frame, imgsz=960, verbose=False,
                                         show=False, save=False)[0]
            barras_raw = sum(resultado_seg.names[int(caja.cls.item())] == "bar"
                             for caja in resultado_seg.boxes)
            caja_banco = seleccionar_banco_frame(resultado_seg, referencia, banco_fijo)
            numero_barras = seleccion_barras(resultado_seg, resultado_pose)

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
                                    texto_banco, numero_barras,
                                    estimaciones.get(frame_indice) if numero_barras == 0 else None,
                                    barras_raw)

            alto, ancho = imagen.shape[:2]
            escala = min(640 / ancho, 360 / alto, 1)
            if escala < 1:
                imagen = cv2.resize(imagen, (round(ancho * escala), round(alto * escala)),
                                    interpolation=cv2.INTER_AREA)
            cv2.imshow(ventana, imagen)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_indice += 1
    finally:
        captura.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
