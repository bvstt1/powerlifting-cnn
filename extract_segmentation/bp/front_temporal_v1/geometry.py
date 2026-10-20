"""Calcula geometría de barra, silueta y pose para features BP front V1.

Todas las operaciones trabajan sobre mascaras crop booleanas + origen,
bboxes en pixeles y keypoints convertidos a pixeles. No se ejecuta YOLO.
"""

import numpy as np

CONF_POSE_MIN = 0.5
WRIST_SEP_MIN_PX = 30.0
SHOULDER_W_MIN_PX = 30.0
GUARD_DY_FRAC_H = 0.03
COV_PERSONA = 0.5
COV_MUNECAS = 0.5
DEDUP_INTER_MIN = 0.85
SLOPE_ANCHO_MIN = 30
SLOPE_COBERTURA_MIN = 0.5
SLOPE_MAX_ABS = 0.35
SLOPE_RESIDUO_FRAC_H = 0.02
PERFIL_ANCHO_MIN = 5
PERFIL_COBERTURA_MIN = 0.7
PERFIL_AMBIG_FRAC_W = 0.35
SEP_COBERTURA_OBS_MIN = 0.6
SEP_ALCANCE_MAX_D_L = 4.0
CONF_ANCLA_BARRA = 0.65

# Indices del modelo frontal de 6 keypoints.
IDX_CODO_IZQ, IDX_HOMBRO_IZQ, IDX_HOMBRO_DER = 0, 1, 2
IDX_CODO_DER, IDX_MUNECA_DER, IDX_MUNECA_IZQ = 3, 4, 5
IDX_HOMBROS = (1, 2)
IDX_MUNECAS = (5, 4)  # izquierda, derecha en orden anatomico


def pose_a_pixeles(pose_t, ancho, alto):
    """pose_t: (6,3) x,y normalizados + conf. Devuelve dict con px y flags."""
    xy = pose_t[:, :2].astype(np.float64)
    conf = pose_t[:, 2].astype(np.float64)
    valido = np.isfinite(xy).all(axis=1) & np.isfinite(conf)
    px = np.full_like(xy, np.nan)
    px[valido, 0] = xy[valido, 0] * ancho
    px[valido, 1] = xy[valido, 1] * alto
    return {"px": px, "conf": conf, "valido": valido}


def munecas_fiables(pose_t, ancho, alto):
    info = pose_a_pixeles(pose_t, ancho, alto)
    px, conf, valido = info["px"], info["conf"], info["valido"]
    for k in IDX_MUNECAS:
        if not valido[k] or conf[k] < CONF_POSE_MIN:
            return None
        if px[k, 0] == 0 and px[k, 1] == 0:
            return None
    m = px[list(IDX_MUNECAS)]
    orden = m[np.argsort(m[:, 0])]
    if orden[1, 0] - orden[0, 0] < WRIST_SEP_MIN_PX:
        return None
    return orden


def hombros_fiables(pose_t, ancho, alto):
    info = pose_a_pixeles(pose_t, ancho, alto)
    px, conf, valido = info["px"], info["conf"], info["valido"]
    for k in IDX_HOMBROS:
        if not valido[k] or conf[k] < CONF_POSE_MIN:
            return None
        if px[k, 0] == 0 and px[k, 1] == 0:
            return None
    return px[list(IDX_HOMBROS)]


def pixeles_globales(mascara_crop, origen):
    """Devuelve (ys, xs) globales a partir del crop bool y origen (x0,y0)."""
    ys_c, xs_c = np.nonzero(mascara_crop)
    if not xs_c.size:
        return ys_c, xs_c
    return ys_c + int(origen[1]), xs_c + int(origen[0])


def interseccion_min(masc_a, org_a, masc_b, org_b):
    """Interseccion / mascara menor entre dos crops (duplicados)."""
    ax0, ay0 = int(org_a[0]), int(org_a[1])
    bx0, by0 = int(org_b[0]), int(org_b[1])
    ah, aw = masc_a.shape
    bh, bw = masc_b.shape
    x0, x1 = max(ax0, bx0), min(ax0 + aw, bx0 + bw)
    y0, y1 = max(ay0, by0), min(ay0 + ah, by0 + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    sub_a = masc_a[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0]
    sub_b = masc_b[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]
    minimo = min(int(masc_a.sum()), int(masc_b.sum()))
    if minimo == 0:
        return 0.0
    return float(np.logical_and(sub_a, sub_b).sum() / minimo)


def seleccionar_barras(instancias, pose_t, ancho, alto):
    """Adapta seleccion_barras del visor. Devuelve dict sin mutar el RAW.

    instancias: lista de dicts {idx_global, clase, conf, bbox, track_id,
        mascara, origen, area}.
    """
    barras = [i for i in instancias if i["clase"] == "bar" and i["mascara"].size]
    personas = [i for i in instancias if i["clase"] == "person" and i["mascara"].size]
    caja_persona = None
    if personas:
        mejor = max(personas, key=lambda i: float(i["conf"]))
        caja_persona = np.asarray(mejor["bbox"], dtype=float)
    mun = munecas_fiables(pose_t, ancho, alto)
    intervalo = (float(mun[0, 0]), float(mun[1, 0])) if mun is not None else None
    altura_manos = float(np.mean(mun[:, 1])) if mun is not None else None

    candidatas = []
    detalle = []
    for b in barras:
        ys, xs = pixeles_globales(b["mascara"], b["origen"])
        if not xs.size:
            detalle.append((b["idx_global"], "mascara_vacia"))
            continue
        if (altura_manos is not None
                and (float(np.median(ys)) - altura_manos) > GUARD_DY_FRAC_H * alto):
            detalle.append((b["idx_global"], "rechazo_inferior_guardia"))
            continue
        dentro = None
        if caja_persona is not None:
            dentro = float(np.mean(
                (xs >= caja_persona[0]) & (xs <= caja_persona[2])
                & (ys >= caja_persona[1]) & (ys <= caja_persona[3])))
        entre = None
        if intervalo is not None:
            entre = float(np.mean((xs >= intervalo[0]) & (xs <= intervalo[1])))
        ok = ((dentro is not None and dentro >= COV_PERSONA)
              or (entre is not None and entre >= COV_MUNECAS))
        detalle.append((b["idx_global"],
                        f"dentro={dentro} entre={entre} ok={ok}"))
        if ok:
            candidatas.append(b)
    candidatas.sort(key=lambda i: float(i["conf"]), reverse=True)
    elegidas = []
    for c in candidatas:
        dup = False
        for e in elegidas:
            if interseccion_min(c["mascara"], c["origen"],
                                e["mascara"], e["origen"]) >= DEDUP_INTER_MIN:
                dup = True
                break
        if not dup:
            elegidas.append(c)
    n_raw = len(barras)
    if n_raw == 0:
        estado = "ausente"
    elif len(elegidas) == 1:
        estado = "asociada"
    elif len(elegidas) > 1:
        estado = "multiple_ambigua"
    else:
        # Hubo detecciones pero ninguna asociada (guardia o cobertura).
        estado = "observada_no_verificada"
    return {"elegidas": elegidas, "n_raw": n_raw,
            "n_kept": len(elegidas), "estado": estado,
            "detalle": detalle, "munecas": mun}


def union_barra(elegidas):
    if not elegidas:
        return None
    return elegidas


def geometria_mascara_union(elegidas, ancho, alto):
    """Centro, tamano y area sobre la union de crops (coords globales)."""
    ys_all, xs_all = [], []
    area = 0
    for b in elegidas:
        ys, xs = pixeles_globales(b["mascara"], b["origen"])
        ys_all.append(ys)
        xs_all.append(xs)
        area += int(b["mascara"].sum())
    if not xs_all or sum(len(x) for x in xs_all) == 0:
        return None
    ys = np.concatenate(ys_all)
    xs = np.concatenate(xs_all)
    return {
        "cx": float(np.median(xs)), "cy": float(np.median(ys)),
        "x_min": int(xs.min()), "x_max": int(xs.max()),
        "y_min": int(ys.min()), "y_max": int(ys.max()),
        "w": int(xs.max() - xs.min()), "h": int(ys.max() - ys.min()),
        "area_px": int(area),
        "area_rel": float(area / (ancho * alto)),
        "ys": ys, "xs": xs,
    }


def inclinacion_barra_union(elegidas, alto):
    """Ajuste del eje visible. Replica umbrales del visor."""
    geo = geometria_mascara_union(elegidas, 1, 1)
    if geo is None:
        return None
    xs, ys = geo["xs"], geo["ys"]
    x_min, x_max = int(xs.min()), int(xs.max())
    ancho_vis = x_max - x_min
    if ancho_vis < SLOPE_ANCHO_MIN:
        return None
    # Agrupar en NumPy por columna: no iterar sobre cada pixel de la mascara.
    orden = np.argsort(xs, kind="stable")
    xs_ord, ys_ord = xs[orden], ys[orden]
    columnas, inicios, largos = np.unique(xs_ord, return_index=True,
                                           return_counts=True)
    centros_col = np.asarray([np.median(ys_ord[a:a + n])
                              for a, n in zip(inicios, largos)])
    centros = []
    for x in range(x_min + int(ancho_vis * 0.15),
                   x_max - int(ancho_vis * 0.15) + 1):
        posicion = np.searchsorted(columnas, x)
        if posicion < len(columnas) and columnas[posicion] == x:
            centros.append((x, float(centros_col[posicion])))
    if len(centros) < ancho_vis * SLOPE_COBERTURA_MIN:
        return None
    valores = np.array(centros)
    pendiente, origen = np.polyfit(valores[:, 0], valores[:, 1], 1)
    residuo = float(np.median(
        np.abs(valores[:, 1] - (pendiente * valores[:, 0] + origen))))
    if abs(float(pendiente)) > SLOPE_MAX_ABS or residuo > alto * SLOPE_RESIDUO_FRAC_H:
        return None
    return {"pendiente": float(pendiente), "origen": float(origen),
            "ancho_visible": int(ancho_vis), "residuo": residuo}


def perfil_central_persona(mascara_crop, origen, x_min, x_max, ancho_frame):
    """Borde superior de silueta por columna. Opera sobre el crop."""
    x_min = max(0, int(x_min))
    x_max = min(ancho_frame - 1, int(x_max))
    if x_max - x_min < PERFIL_ANCHO_MIN:
        return None
    x0, y0 = int(origen[0]), int(origen[1])
    h, w = mascara_crop.shape
    perfil = []
    for x in range(x_min, x_max + 1):
        xc = x - x0
        if xc < 0 or xc >= w:
            continue
        col = np.flatnonzero(mascara_crop[:, xc])
        if col.size:
            perfil.append((x, int(col[0] + y0)))
    if len(perfil) < PERFIL_COBERTURA_MIN * (x_max - x_min + 1):
        return None
    return (int(np.median([x for x, _ in perfil])),
            int(np.median([y for _, y in perfil])), perfil)


def zona_central(pose_t, persona_inst, banco_bbox, ancho, alto):
    """Franja entre hombros (pose) o respaldo por banco. Replica visor."""
    if persona_inst is None:
        return None
    hombros = hombros_fiables(pose_t, ancho, alto)
    if hombros is not None:
        (x1, y1), (x2, y2) = hombros
        x_izq, x_der = sorted((float(x1), float(x2)))
        w = x_der - x_izq
        if w >= SHOULDER_W_MIN_PX:
            perfil = perfil_central_persona(
                persona_inst["mascara"], persona_inst["origen"],
                x_izq + 0.25 * w, x_izq + 0.75 * w, ancho)
            if perfil is not None:
                ambiguo = min(float(y1), float(y2)) - perfil[1] > PERFIL_AMBIG_FRAC_W * w
                return {"perfil": perfil, "origen": "pose",
                        "ambiguo": bool(ambiguo), "ancho_hombros": float(w)}
    if banco_bbox is None:
        return None
    x1, _, x2, _ = [float(v) for v in banco_bbox]
    w = x2 - x1
    if w < 0.1 * ancho:
        return None
    centro = (x1 + x2) / 2
    perfil = perfil_central_persona(
        persona_inst["mascara"], persona_inst["origen"],
        centro - 0.2 * w, centro + 0.2 * w, ancho)
    if perfil is not None:
        return {"perfil": perfil, "origen": "banco",
                "ambiguo": True, "ancho_hombros": None}
    return None


def separacion_barra_silueta(elegidas, perfil_persona, alto, ancho):
    """Devuelve (sep_px, puntos_barra, proyectado, cobertura, dist_extra, ancho_vis).

    sep = mediana(y_persona - y_barra): + barra por encima (separada).
    """
    if not elegidas or perfil_persona is None:
        return None
    xs_list = [(x, y) for x, y in perfil_persona]
    ys_all = np.concatenate(
        [pixeles_globales(b["mascara"], b["origen"])[0] for b in elegidas])
    xs_all = np.concatenate(
        [pixeles_globales(b["mascara"], b["origen"])[1] for b in elegidas])
    bordes = np.full(int(xs_all.max()) + 1, -1, dtype=np.int32)
    np.maximum.at(bordes, xs_all, ys_all)
    observado = []
    for x, y_p in xs_list:
        if 0 <= x < len(bordes) and bordes[x] >= 0:
            observado.append((x, int(bordes[x]), y_p))
    if len(observado) >= SEP_COBERTURA_OBS_MIN * len(xs_list):
        seps = [yp - yb for _, yb, yp in observado]
        return {"sep_px": int(np.median(seps)),
                "puntos": [(x, y) for x, y, _ in observado],
                "proyectado": False,
                "cobertura": float(len(observado) / len(xs_list)),
                "dist_extra": 0.0, "ancho_visible": None}
    ajuste = inclinacion_barra_union(elegidas, alto)
    if ajuste is None:
        return None
    pendiente, ancho_vis = ajuste["pendiente"], ajuste["ancho_visible"]
    x_centro = int(np.median(xs_all))
    franja = np.abs(xs_all - x_centro) <= max(4, int(ancho_vis * 0.1))
    if not np.any(franja):
        return None
    y_borde = float(np.percentile(ys_all[franja], 85))
    x_obj = int(np.median([x for x, _ in xs_list]))
    distancia = float(min(abs(x_obj - xs_all.min()), abs(x_obj - xs_all.max())))
    if distancia > SEP_ALCANCE_MAX_D_L * ancho_vis:
        return None
    puntos = [(x, round(y_borde + pendiente * (x - x_centro)))
              for x, _ in xs_list]
    seps = [yp - yb for (_, yp), (_, yb) in zip(xs_list, puntos)]
    return {"sep_px": int(np.median(seps)), "puntos": puntos,
            "proyectado": True,
            "cobertura": float(len(observado) / len(xs_list)) if xs_list else 0.0,
            "dist_extra": distancia, "ancho_visible": int(ancho_vis)}
